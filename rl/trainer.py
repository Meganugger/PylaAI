# PPO trainer: collects experience, trains small MLP policy between matches

from __future__ import annotations

import math
import os
import time
import json
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Try to import torch -- graceful fallback if not installed
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[RL] PyTorch not found -- RL training disabled. Install with: pip install torch")

# Try to import torch-directml for AMD GPU support
DIRECTML_AVAILABLE = False
if TORCH_AVAILABLE:
    try:
        import torch_directml
        DIRECTML_AVAILABLE = True
        print("[RL] DirectML (AMD GPU) support available")
    except ImportError:
        pass


# action Space
# Discrete action space for the bot
MOVEMENT_ACTIONS = [
    "",     # No movement (stand still)
    "W",    # Up
    "S",    # Down
    "A",    # Left
    "D",    # Right
    "WA",   # Up-Left
    "WD",   # Up-Right
    "SA",   # Down-Left
    "SD",   # Down-Right
]

ATTACK_ACTIONS = [
    "none",         # Don't attack
    "auto_aim",     # Auto-aim attack (M key)
    "aimed",        # Aimed attack (drag toward target)
]

ABILITY_ACTIONS = [
    "none",         # No ability
    "super",        # Use super
    "gadget",       # Use gadget
    "hypercharge",  # Use hypercharge
]

N_MOVEMENT = len(MOVEMENT_ACTIONS)     # 9
N_ATTACK = len(ATTACK_ACTIONS)         # 3
N_ABILITY = len(ABILITY_ACTIONS)       # 4

# Flat action index (kept for backward compat with stored transitions):
# movement * (N_ATTACK * N_ABILITY) + attack * N_ABILITY + ability
TOTAL_ACTIONS = N_MOVEMENT * N_ATTACK * N_ABILITY  # 9 × 3 × 4 = 108


def decode_action(action_idx) -> Tuple[str, str, str]:
    """Decode action(s) into (movement, attack, ability) strings.

    Accepts either:
      - int: legacy flat index (0..107)
      - tuple/list of 3 ints: multi-head indices (move_idx, atk_idx, abi_idx)
    """
    if isinstance(action_idx, (list, tuple)):
        m, a, ab = action_idx
        return (
            MOVEMENT_ACTIONS[min(int(m), N_MOVEMENT - 1)],
            ATTACK_ACTIONS[min(int(a), N_ATTACK - 1)],
            ABILITY_ACTIONS[min(int(ab), N_ABILITY - 1)],
        )
    # Legacy flat index
    ability_idx = action_idx % N_ABILITY
    attack_idx = (action_idx // N_ABILITY) % N_ATTACK
    movement_idx = action_idx // (N_ATTACK * N_ABILITY)

    return (
        MOVEMENT_ACTIONS[min(movement_idx, N_MOVEMENT - 1)],
        ATTACK_ACTIONS[min(attack_idx, N_ATTACK - 1)],
        ABILITY_ACTIONS[min(ability_idx, N_ABILITY - 1)],
    )


def encode_action(movement: str, attack: str, ability: str) -> Tuple[int, int, int]:
    """Encode (movement, attack, ability) into per-head action indices.

    Returns: (move_idx, attack_idx, ability_idx)
    """
    m_idx = MOVEMENT_ACTIONS.index(movement) if movement in MOVEMENT_ACTIONS else 0
    a_idx = ATTACK_ACTIONS.index(attack) if attack in ATTACK_ACTIONS else 0
    ab_idx = ABILITY_ACTIONS.index(ability) if ability in ABILITY_ACTIONS else 0
    return (m_idx, a_idx, ab_idx)


def encode_action_flat(movement: str, attack: str, ability: str) -> int:
    """Encode (movement, attack, ability) into a flat action index (legacy)."""
    m_idx = MOVEMENT_ACTIONS.index(movement) if movement in MOVEMENT_ACTIONS else 0
    a_idx = ATTACK_ACTIONS.index(attack) if attack in ATTACK_ACTIONS else 0
    ab_idx = ABILITY_ACTIONS.index(ability) if ability in ABILITY_ACTIONS else 0
    return m_idx * (N_ATTACK * N_ABILITY) + a_idx * N_ABILITY + ab_idx


# experience Buffer
class ExperienceBuffer:
    """Stores transitions for PPO training.

    Multi-head format:
      action = (move_idx, attack_idx, ability_idx)  — 3 ints
      log_prob = (move_lp, attack_lp, ability_lp)   — 3 floats
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.states = deque(maxlen=max_size)
        self.move_actions = deque(maxlen=max_size)
        self.attack_actions = deque(maxlen=max_size)
        self.ability_actions = deque(maxlen=max_size)
        self.rewards = deque(maxlen=max_size)
        self.dones = deque(maxlen=max_size)
        self.log_probs = deque(maxlen=max_size)   # sum of 3 head log probs
        self.values = deque(maxlen=max_size)

    def add(self, state: np.ndarray, action, reward: float,
            done: bool, log_prob, value: float):
        """Add a transition.

        action: (move_idx, attack_idx, ability_idx) or legacy int
        log_prob: (move_lp, attack_lp, ability_lp) or legacy float
        """
        self.states.append(state)

        # Handle both multi-head tuple and legacy flat int
        if isinstance(action, (list, tuple)) and len(action) == 3:
            self.move_actions.append(action[0])
            self.attack_actions.append(action[1])
            self.ability_actions.append(action[2])
        else:
            # Legacy flat action → decompose
            a = int(action)
            self.ability_actions.append(a % N_ABILITY)
            self.attack_actions.append((a // N_ABILITY) % N_ATTACK)
            self.move_actions.append(a // (N_ATTACK * N_ABILITY))

        self.rewards.append(reward)
        self.dones.append(done)

        # Sum log probs for GAE computation
        if isinstance(log_prob, (list, tuple)):
            self.log_probs.append(sum(log_prob))
        else:
            self.log_probs.append(float(log_prob))

        self.values.append(value)

    def get_batch(self) -> Optional[Dict[str, Any]]:
        """Get all stored transitions as numpy arrays."""
        if len(self.states) < 2:
            return None
        return {
            "states": np.array(self.states, dtype=np.float32),
            "move_actions": np.array(self.move_actions, dtype=np.int64),
            "attack_actions": np.array(self.attack_actions, dtype=np.int64),
            "ability_actions": np.array(self.ability_actions, dtype=np.int64),
            "rewards": np.array(self.rewards, dtype=np.float32),
            "dones": np.array(self.dones, dtype=np.float32),
            "log_probs": np.array(self.log_probs, dtype=np.float32),
            "values": np.array(self.values, dtype=np.float32),
        }

    def clear(self):
        self.states.clear()
        self.move_actions.clear()
        self.attack_actions.clear()
        self.ability_actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def __len__(self):
        return len(self.states)


# neural Network (Multi-Head Policy + Value)
if TORCH_AVAILABLE:
    class ActorCritic(nn.Module):
        """Multi-head actor-critic network for PPO.

        3 independent policy heads instead of 1 flat 108-action head:
          - Movement head:  9 actions  (stand, W, S, A, D, WA, WD, SA, SD)
          - Attack head:    3 actions  (none, auto_aim, aimed)
          - Ability head:   4 actions  (none, super, gadget, hypercharge)

        Each head only needs to learn its own small action space.
        Total effective complexity: 9 + 3 + 4 = 16 instead of 108.

        Architecture:
          Input -> 256 -> 256 -> 128 -> (move_head, attack_head, ability_head, value_head)
        """

        def __init__(self, state_dim: int, action_dim: int = TOTAL_ACTIONS):
            super().__init__()

            # Shared backbone (identical to old architecture)
            self.shared = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
            )

            # 3 independent policy heads
            self.move_head = nn.Sequential(
                nn.Linear(128, N_MOVEMENT),
                nn.Softmax(dim=-1),
            )
            self.attack_head = nn.Sequential(
                nn.Linear(128, N_ATTACK),
                nn.Softmax(dim=-1),
            )
            self.ability_head = nn.Sequential(
                nn.Linear(128, N_ABILITY),
                nn.Softmax(dim=-1),
            )

            # Value head (critic) — unchanged
            self.value = nn.Sequential(
                nn.Linear(128, 1),
            )

        def forward(self, state):
            shared = self.shared(state)
            move_probs = self.move_head(shared)
            attack_probs = self.attack_head(shared)
            ability_probs = self.ability_head(shared)
            value = self.value(shared)
            return move_probs, attack_probs, ability_probs, value

        def get_action(self, state: np.ndarray) -> Tuple[Tuple[int, int, int], Tuple[float, float, float], float]:
            """Select action using multi-head policy.

            Returns:
              actions: (move_idx, attack_idx, ability_idx)
              log_probs: (move_lp, attack_lp, ability_lp)
              value: float
            """
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                move_p, attack_p, ability_p, value = self(state_tensor)

                move_dist = Categorical(move_p)
                attack_dist = Categorical(attack_p)
                ability_dist = Categorical(ability_p)

                move_a = move_dist.sample()
                attack_a = attack_dist.sample()
                ability_a = ability_dist.sample()

                move_lp = move_dist.log_prob(move_a)
                attack_lp = attack_dist.log_prob(attack_a)
                ability_lp = ability_dist.log_prob(ability_a)

            actions = (move_a.item(), attack_a.item(), ability_a.item())
            log_probs = (move_lp.item(), attack_lp.item(), ability_lp.item())
            return actions, log_probs, value.item()

        def evaluate(self, states, actions_tuple):
            """Evaluate actions for PPO loss computation.

            Args:
              states: [B, state_dim]
              actions_tuple: (move_actions[B], attack_actions[B], ability_actions[B])

            Returns: (log_probs_sum[B], values[B], entropy_sum[B])
            """
            move_p, attack_p, ability_p, values = self(states)

            move_acts, attack_acts, ability_acts = actions_tuple

            move_dist = Categorical(move_p)
            attack_dist = Categorical(attack_p)
            ability_dist = Categorical(ability_p)

            # Sum log probs (joint = product → log = sum)
            log_probs = (move_dist.log_prob(move_acts) +
                         attack_dist.log_prob(attack_acts) +
                         ability_dist.log_prob(ability_acts))

            # Sum entropies for exploration bonus
            entropy = (move_dist.entropy() +
                       attack_dist.entropy() +
                       ability_dist.entropy())

            return log_probs, values.squeeze(-1), entropy


# pPO Trainer
class PPOTrainer:
    """Proximal Policy Optimization trainer.

    Handles:
      - Action selection during gameplay
      - Experience collection
      - Policy updates between/during matches
      - Model saving/loading
    """

    def __init__(self, state_dim: int, action_dim: int = TOTAL_ACTIONS,
                 lr: float = 3e-4, gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2, value_coef: float = 0.5,
                 value_clip: float = 0.2,
                 entropy_coef: float = 0.01, max_grad_norm: float = 0.5,
                 update_epochs: int = 4, batch_size: int = 64,
                 buffer_size: int = 4096, model_dir: str = "rl_models"):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.value_clip = value_clip
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.model_dir = model_dir

        self.buffer = ExperienceBuffer(buffer_size)

        # Training stats
        self.total_updates = 0
        self.total_episodes = 0
        self.episode_rewards: deque = deque(maxlen=500)
        self.avg_reward_window = deque(maxlen=50)

        # --- Extended stats for training_stats.json overview ---
        self.total_wins = 0
        self.total_losses = 0
        self.total_kills = 0
        self.total_deaths = 0
        self.total_damage_dealt = 0.0
        self.total_damage_taken = 0.0
        self.total_overextend_events = 0
        self.total_low_hp_engage_events = 0
        self.total_rl_move_overrides = 0
        self.total_rl_move_override_blocks = 0
        self.total_rl_override_block_attack_window = 0
        self.total_rl_override_block_pattern_pressure = 0
        self.total_rl_override_critical_applied = 0
        self.total_peek_active_frames = 0
        self.total_water_pressure_frames = 0
        self.total_enemy_pattern_pressure_frames = 0
        self.total_enemy_attack_soon_frames = 0
        self.total_kpi_reward_adjustment = 0.0
        self.best_reward = -999999.0
        self.worst_reward = 999999.0
        self.best_kill_streak = 0
        self.training_start_time: float = time.time()
        self._cumulative_training_hours: float = 0.0   # Restored from JSON
        self.last_update_time: float = 0.0

        # Per-episode detailed history (last 100 episodes)
        self.episode_history: List[Dict[str, Any]] = []
        self._max_episode_history = 100

        # Training loss history (last 100 updates)
        self.loss_history: List[Dict[str, float]] = []
        self._max_loss_history = 100

        # Per-brawler breakdown
        self.brawler_stats: Dict[str, Dict[str, Any]] = {}

        # Rolling windows for trend detection
        self._reward_window_10 = deque(maxlen=10)
        self._reward_window_50 = deque(maxlen=50)
        self._kill_window_10 = deque(maxlen=10)
        self._winrate_window_20 = deque(maxlen=20)
        self._recent_damage_trade_50 = deque(maxlen=50)
        self._recent_death_flag_50 = deque(maxlen=50)
        self._last_reward_usage_rates: Dict[str, float] = {}
        self._last_disabled_reward_signals: Dict[str, float] = {}

        if TORCH_AVAILABLE:
            # Smart device selection: DirectML (AMD) > CUDA (NVIDIA) > CPU
            if DIRECTML_AVAILABLE:
                self.device = torch_directml.device()
                print(f"[RL] Using DirectML (AMD GPU) for training")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"[RL] Using CUDA (NVIDIA GPU) for training")
            else:
                self.device = torch.device("cpu")
                print(f"[RL] Using CPU for training")
            self.model = ActorCritic(state_dim, action_dim).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
            self.model.eval()  # Default to eval mode
        else:
            self.model = None
            self.optimizer = None

    @property
    def is_available(self) -> bool:
        """Whether RL training is available (PyTorch installed)."""
        return TORCH_AVAILABLE and self.model is not None

    def select_action(self, state: np.ndarray, explore: bool = True):
        """Select an action given the current state (multi-head).

        Returns:
          actions: (move_idx, attack_idx, ability_idx)
          log_probs: (move_lp, attack_lp, ability_lp)
          value: float
        """
        if not self.is_available:
            return (0, 0, 0), (0.0, 0.0, 0.0), 0.0

        if explore:
            return self.model.get_action(state)
        else:
            # Greedy: pick argmax from each head
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                move_p, attack_p, ability_p, value = self.model(state_tensor)
                actions = (
                    move_p.argmax(dim=-1).item(),
                    attack_p.argmax(dim=-1).item(),
                    ability_p.argmax(dim=-1).item(),
                )
                # Compute log probs for the greedy actions
                move_lp = torch.log(move_p[0, actions[0]] + 1e-8).item()
                attack_lp = torch.log(attack_p[0, actions[1]] + 1e-8).item()
                ability_lp = torch.log(ability_p[0, actions[2]] + 1e-8).item()
            return actions, (move_lp, attack_lp, ability_lp), value.item()

    def store_transition(self, state: np.ndarray, action, reward: float,
                          done: bool, log_prob, value: float):
        """Store a transition in the experience buffer.

        action: (move_idx, attack_idx, ability_idx) or legacy int
        log_prob: (move_lp, attack_lp, ability_lp) or legacy float
        """
        self.buffer.add(state, action, reward, done, log_prob, value)

    def update(self) -> Optional[Dict[str, float]]:
        """Run PPO update using collected experience (multi-head).

        Returns: Training metrics dict, or None if not enough data.
        """
        if not self.is_available:
            return None

        batch = self.buffer.get_batch()
        if batch is None or len(batch["states"]) < self.batch_size:
            return None

        self.model.train()

        states = torch.FloatTensor(batch["states"]).to(self.device)
        move_actions = torch.LongTensor(batch["move_actions"]).to(self.device)
        attack_actions = torch.LongTensor(batch["attack_actions"]).to(self.device)
        ability_actions = torch.LongTensor(batch["ability_actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(batch["log_probs"]).to(self.device)
        rewards = batch["rewards"]
        dones = batch["dones"]
        values = batch["values"]

        # Compute GAE (Generalized Advantage Estimation)
        advantages = self._compute_gae(rewards, values, dones)
        returns = advantages + values

        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        if len(advantages_t) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # PPO update epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        n = len(states)
        for epoch in range(self.update_epochs):
            # Mini-batch training
            indices = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]
                mb_actions_tuple = (
                    move_actions[mb_idx],
                    attack_actions[mb_idx],
                    ability_actions[mb_idx],
                )
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages_t[mb_idx]
                mb_returns = returns_t[mb_idx]
                mb_old_values = torch.FloatTensor(values[mb_idx]).to(self.device)

                # Evaluate current policy (multi-head)
                log_probs, value_preds, entropy = self.model.evaluate(
                    mb_states, mb_actions_tuple)

                # PPO clipped objective
                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (PPO-style value clipping)
                value_pred_clipped = mb_old_values + torch.clamp(
                    value_preds - mb_old_values,
                    -self.value_clip,
                    self.value_clip,
                )
                value_losses = (value_preds - mb_returns).pow(2)
                value_losses_clipped = (value_pred_clipped - mb_returns).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (policy_loss +
                        self.value_coef * value_loss +
                        self.entropy_coef * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()

        self.model.eval()
        self.total_updates += 1

        # Clear buffer after update
        self.buffer.clear()

        n_batches = max(1, (n * self.update_epochs) // self.batch_size)
        metrics = {
            "policy_loss": total_policy_loss / n_batches,
            "value_loss": total_value_loss / n_batches,
            "entropy": total_entropy / n_batches,
            "buffer_size": n,
            "updates": self.total_updates,
        }

        # Store loss history for trend analysis
        self.last_update_time = time.time()
        loss_record = {
            "update": self.total_updates,
            "policy_loss": round(metrics["policy_loss"], 6),
            "value_loss": round(metrics["value_loss"], 6),
            "entropy": round(metrics["entropy"], 4),
            "buffer_size": n,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.loss_history.append(loss_record)
        if len(self.loss_history) > self._max_loss_history:
            self.loss_history.pop(0)

        return metrics

    def _compute_gae(self, rewards: np.ndarray, values: np.ndarray,
                      dones: np.ndarray) -> np.ndarray:
        """Compute Generalized Advantage Estimation."""
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0

        for t in reversed(range(n - 1)):
            next_value = values[t + 1] * (1 - dones[t + 1])
            delta = rewards[t] + self.gamma * next_value - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t + 1]) * last_gae
            advantages[t] = last_gae

        return advantages

    def end_episode(self, total_reward: float, summary: Optional[Dict[str, Any]] = None,
                    brawler: str = "unknown", match_won: bool = False):
        """Called at end of each match/episode for tracking.

        Args:
            total_reward: Final cumulative reward for this episode.
            summary:  Episode summary dict from RewardCalculator.get_episode_summary().
            brawler:  Name of the brawler used (for per-brawler stats).
            match_won: Whether the match was won.
        """
        self.total_episodes += 1
        self.episode_rewards.append(total_reward)
        self.avg_reward_window.append(total_reward)
        self._reward_window_10.append(total_reward)
        self._reward_window_50.append(total_reward)

        # Track best/worst
        if total_reward > self.best_reward:
            self.best_reward = total_reward
        if total_reward < self.worst_reward:
            self.worst_reward = total_reward

        # Win/loss counters
        if match_won:
            self.total_wins += 1
            self._winrate_window_20.append(1)
        else:
            self.total_losses += 1
            self._winrate_window_20.append(0)

        # Extract per-match stats from summary
        kills = 0
        deaths = 0
        damage_dealt = 0.0
        damage_taken = 0.0
        accuracy = 0.0
        best_streak = 0
        dodges = 0
        overextend_events = 0
        low_hp_engage_events = 0
        rl_move_override_applied = 0
        rl_move_override_blocked = 0
        rl_move_override_blocked_attack_window = 0
        rl_move_override_blocked_pattern_pressure = 0
        rl_move_override_critical_applied = 0
        peek_active_frames = 0
        water_pressure_frames = 0
        enemy_pattern_pressure_frames = 0
        enemy_attack_soon_frames = 0
        kpi_reward_adjustment = 0.0
        kpi_profile = "unknown"
        if summary:
            kills = summary.get("kills", 0)
            deaths = summary.get("deaths", 0)
            damage_dealt = summary.get("damage_dealt", 0.0)
            damage_taken = summary.get("damage_taken", 0.0)
            accuracy = max(0.0, min(1.0, float(summary.get("accuracy", 0.0) or 0.0)))
            best_streak = summary.get("kill_streak", 0)
            dodges = summary.get("dodges", 0)
            overextend_events = int(summary.get("overextend_events", 0) or 0)
            low_hp_engage_events = int(summary.get("low_hp_engage_events", 0) or 0)
            rl_move_override_applied = int(summary.get("rl_move_override_applied", 0) or 0)
            rl_move_override_blocked = int(summary.get("rl_move_override_blocked", 0) or 0)
            rl_move_override_blocked_attack_window = int(summary.get("rl_move_override_blocked_attack_window", 0) or 0)
            rl_move_override_blocked_pattern_pressure = int(summary.get("rl_move_override_blocked_pattern_pressure", 0) or 0)
            rl_move_override_critical_applied = int(summary.get("rl_move_override_critical_applied", 0) or 0)
            peek_active_frames = int(summary.get("peek_active_frames", 0) or 0)
            water_pressure_frames = int(summary.get("water_pressure_frames", 0) or 0)
            enemy_pattern_pressure_frames = int(summary.get("enemy_pattern_pressure_frames", 0) or 0)
            enemy_attack_soon_frames = int(summary.get("enemy_attack_soon_frames", 0) or 0)
            kpi_reward_adjustment = float(summary.get("kpi_reward_adjustment", 0.0) or 0.0)
            kpi_profile = str(summary.get("kpi_profile", "unknown") or "unknown").lower()
            usage_rates = summary.get("reward_usage_rates", {})
            if isinstance(usage_rates, dict):
                self._last_reward_usage_rates = {
                    str(k): float(v) for k, v in usage_rates.items()
                }
            disabled_signals = summary.get("disabled_reward_signals", {})
            if isinstance(disabled_signals, dict):
                self._last_disabled_reward_signals = {
                    str(k): float(v) for k, v in disabled_signals.items()
                }

        self.total_kills += kills
        self.total_deaths += deaths
        self.total_damage_dealt += damage_dealt
        self.total_damage_taken += damage_taken
        self.total_overextend_events += overextend_events
        self.total_low_hp_engage_events += low_hp_engage_events
        self.total_rl_move_overrides += rl_move_override_applied
        self.total_rl_move_override_blocks += rl_move_override_blocked
        self.total_rl_override_block_attack_window += rl_move_override_blocked_attack_window
        self.total_rl_override_block_pattern_pressure += rl_move_override_blocked_pattern_pressure
        self.total_rl_override_critical_applied += rl_move_override_critical_applied
        self.total_peek_active_frames += peek_active_frames
        self.total_water_pressure_frames += water_pressure_frames
        self.total_enemy_pattern_pressure_frames += enemy_pattern_pressure_frames
        self.total_enemy_attack_soon_frames += enemy_attack_soon_frames
        self.total_kpi_reward_adjustment += kpi_reward_adjustment
        self.best_kill_streak = max(self.best_kill_streak, best_streak)
        self._kill_window_10.append(kills)
        self._recent_damage_trade_50.append(float(damage_dealt) / max(1.0, float(damage_taken)))
        self._recent_death_flag_50.append(1 if deaths > 0 else 0)

        # Build episode record
        episode_record = {
            "episode": self.total_episodes,
            "reward": round(total_reward, 2),
            "won": match_won,
            "brawler": brawler,
            "kills": kills,
            "deaths": deaths,
            "kda": kills - deaths,
            "damage_dealt": int(damage_dealt),
            "damage_taken": int(damage_taken),
            "accuracy": round(accuracy, 3),
            "kill_streak": best_streak,
            "dodges": dodges,
            "overextend_events": overextend_events,
            "low_hp_engage_events": low_hp_engage_events,
            "rl_move_override_applied": rl_move_override_applied,
            "rl_move_override_blocked": rl_move_override_blocked,
            "rl_move_override_blocked_attack_window": rl_move_override_blocked_attack_window,
            "rl_move_override_blocked_pattern_pressure": rl_move_override_blocked_pattern_pressure,
            "rl_move_override_critical_applied": rl_move_override_critical_applied,
            "peek_active_frames": peek_active_frames,
            "water_pressure_frames": water_pressure_frames,
            "enemy_pattern_pressure_frames": enemy_pattern_pressure_frames,
            "enemy_attack_soon_frames": enemy_attack_soon_frames,
            "kpi_reward_adjustment": round(kpi_reward_adjustment, 4),
            "kpi_profile": kpi_profile,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.episode_history.append(episode_record)
        if len(self.episode_history) > self._max_episode_history:
            self.episode_history.pop(0)

        # Per-brawler stats
        if brawler not in self.brawler_stats:
            self.brawler_stats[brawler] = {
                "matches": 0, "wins": 0, "losses": 0,
                "kills": 0, "deaths": 0,
                "total_reward": 0.0, "damage_dealt": 0.0,
            }
        bs = self.brawler_stats[brawler]
        bs["matches"] += 1
        bs["wins"] += int(match_won)
        bs["losses"] += int(not match_won)
        bs["kills"] += kills
        bs["deaths"] += deaths
        bs["total_reward"] += total_reward
        bs["damage_dealt"] += damage_dealt

    def save(self, path: Optional[str] = None):
        """Save model weights and training state."""
        if not self.is_available:
            return

        save_dir = path or self.model_dir
        os.makedirs(save_dir, exist_ok=True)

        model_path = os.path.join(save_dir, "policy.pt")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_updates": self.total_updates,
            "total_episodes": self.total_episodes,
        }, model_path)

        # --- Build comprehensive training stats ---
        avg_50 = float(np.mean(list(self._reward_window_50))) if self._reward_window_50 else 0.0
        avg_10 = float(np.mean(list(self._reward_window_10))) if self._reward_window_10 else 0.0
        winrate_20 = (sum(self._winrate_window_20) / max(1, len(self._winrate_window_20))
                      * 100) if self._winrate_window_20 else 0.0
        avg_kills_10 = float(np.mean(list(self._kill_window_10))) if self._kill_window_10 else 0.0
        tracked_matches = self.total_wins + self.total_losses
        global_winrate = (self.total_wins / max(1, tracked_matches)) * 100
        global_kda = (self.total_kills - self.total_deaths) / max(1, tracked_matches)
        training_hours = self._cumulative_training_hours + (time.time() - self.training_start_time) / 3600

        # Trend: compare recent 10 vs the 10 before that
        # Use episode_history rewards which survive restarts (not just window)
        recent_rewards = [ep["reward"] for ep in self.episode_history[-20:]]
        if len(recent_rewards) >= 20:
            old_avg = float(np.mean(recent_rewards[:10]))
            new_avg = float(np.mean(recent_rewards[10:]))
            trend = "improving" if new_avg > old_avg + 1 else "declining" if new_avg < old_avg - 1 else "stable"
            trend_delta = round(new_avg - old_avg, 2)
        elif len(recent_rewards) >= 10:
            # At least 10 matches: compare first half vs second half
            mid = len(recent_rewards) // 2
            old_avg = float(np.mean(recent_rewards[:mid]))
            new_avg = float(np.mean(recent_rewards[mid:]))
            trend = "improving" if new_avg > old_avg + 2 else "declining" if new_avg < old_avg - 2 else "stable"
            trend_delta = round(new_avg - old_avg, 2)
        else:
            trend = "insufficient_data"
            trend_delta = 0.0

        # Per-brawler summary with win rate
        brawler_summary = {}
        for name, bs in self.brawler_stats.items():
            brawler_summary[name] = {
                "matches": bs["matches"],
                "wins": bs["wins"],
                "losses": bs["losses"],
                "win_rate": f"{bs['wins'] / max(1, bs['matches']) * 100:.1f}%",
                "kills": bs["kills"],
                "deaths": bs["deaths"],
                "avg_reward": round(bs["total_reward"] / max(1, bs["matches"]), 2),
                "avg_damage": int(bs["damage_dealt"] / max(1, bs["matches"])),
            }

        recent_matches = self.episode_history[-50:]
        recent_20 = self.episode_history[-20:]
        recent_10 = self.episode_history[-10:]

        recent_kills = sum(ep.get("kills", 0) for ep in recent_20)
        recent_deaths = sum(ep.get("deaths", 0) for ep in recent_20)
        recent_kd = recent_kills / max(1, recent_deaths)

        death_rate_20 = (
            sum(1 for ep in recent_20 if ep.get("deaths", 0) > 0) / max(1, len(recent_20))
        )
        avg_damage_trade_20 = (
            float(np.mean([ep.get("damage_dealt", 0) / max(1, ep.get("damage_taken", 0)) for ep in recent_20]))
            if recent_20 else 0.0
        )
        overextend_rate_20 = (
            float(np.mean([ep.get("overextend_events", 0) for ep in recent_20]))
            if recent_20 else 0.0
        )
        low_hp_engage_rate_20 = (
            float(np.mean([ep.get("low_hp_engage_events", 0) for ep in recent_20]))
            if recent_20 else 0.0
        )
        rl_override_apply_rate_20 = (
            float(np.mean([ep.get("rl_move_override_applied", 0) for ep in recent_20]))
            if recent_20 else 0.0
        )
        rl_override_block_rate_20 = (
            float(np.mean([ep.get("rl_move_override_blocked", 0) for ep in recent_20]))
            if recent_20 else 0.0
        )
        rl_override_block_attack_window_rate_20 = (
            float(np.mean([ep.get("rl_move_override_blocked_attack_window", 0) for ep in recent_20]))
            if recent_20 else 0.0
        )
        rl_override_block_pattern_pressure_rate_20 = (
            float(np.mean([ep.get("rl_move_override_blocked_pattern_pressure", 0) for ep in recent_20]))
            if recent_20 else 0.0
        )
        rl_override_critical_applied_rate_20 = (
            float(np.mean([ep.get("rl_move_override_critical_applied", 0) for ep in recent_20]))
            if recent_20 else 0.0
        )
        peek_active_rate_20 = (
            float(np.mean([ep.get("peek_active_frames", 0) for ep in recent_20]))
            if recent_20 else 0.0
        )
        water_pressure_rate_20 = (
            float(np.mean([ep.get("water_pressure_frames", 0) for ep in recent_20]))
            if recent_20 else 0.0
        )
        enemy_pattern_pressure_rate_20 = (
            float(np.mean([ep.get("enemy_pattern_pressure_frames", 0) for ep in recent_20]))
            if recent_20 else 0.0
        )
        enemy_attack_soon_rate_20 = (
            float(np.mean([ep.get("enemy_attack_soon_frames", 0) for ep in recent_20]))
            if recent_20 else 0.0
        )
        kpi_reward_adjustment_20 = (
            float(np.mean([ep.get("kpi_reward_adjustment", 0.0) for ep in recent_20]))
            if recent_20 else 0.0
        )
        kpi_profile_distribution_20 = {}
        if recent_20:
            for ep in recent_20:
                profile_name = str(ep.get("kpi_profile", "unknown") or "unknown").lower()
                kpi_profile_distribution_20[profile_name] = kpi_profile_distribution_20.get(profile_name, 0) + 1

        one_k_one_d_eps = [ep for ep in recent_matches if ep.get("kills", 0) == 1 and ep.get("deaths", 0) == 1]
        one_k_one_d_negative = (
            float(np.mean([ep.get("reward", 0.0) for ep in one_k_one_d_eps])) < 0.0
            if one_k_one_d_eps else None
        )

        equal_trade_eps = [
            ep for ep in recent_matches
            if 0.9 <= (ep.get("damage_dealt", 0) / max(1, ep.get("damage_taken", 0))) <= 1.1
        ]
        equal_trade_negative = (
            float(np.mean([ep.get("reward", 0.0) for ep in equal_trade_eps])) < 0.0
            if equal_trade_eps else None
        )

        value_losses = [lh.get("value_loss", 0.0) for lh in self.loss_history[-20:] if isinstance(lh, dict)]
        value_spike_count = 0
        value_loss_stable = None
        if value_losses:
            p50 = float(np.median(value_losses))
            spike_threshold = max(2.0, p50 * 3.0)
            value_spike_count = sum(1 for v in value_losses if v > spike_threshold)
            value_loss_stable = value_spike_count <= 1

        passive_hide_risk = None
        if recent_10:
            avg_kills_10 = float(np.mean([ep.get("kills", 0) for ep in recent_10]))
            avg_dd_10 = float(np.mean([ep.get("damage_dealt", 0) for ep in recent_10]))
            avg_dt_10 = float(np.mean([ep.get("damage_taken", 0) for ep in recent_10]))
            passive_hide_risk = (avg_kills_10 < 0.2 and avg_dd_10 < 2000 and avg_dt_10 < 2000)

        top_reward_usage = sorted(
            self._last_reward_usage_rates.items(), key=lambda kv: kv[1], reverse=True
        )[:10]

        stats = {
            # === OVERVIEW ===
            "overview": {
                "total_episodes": self.total_episodes,
                "total_updates": self.total_updates,
                "training_hours": round(training_hours, 2),
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            # === REWARDS ===
            "rewards": {
                "avg_reward_last_10": round(avg_10, 2),
                "avg_reward_last_50": round(avg_50, 2),
                "best_reward": round(self.best_reward, 2),
                "worst_reward": round(self.worst_reward, 2),
                "trend": trend,
                "trend_delta": trend_delta,
                "last_10_rewards": [round(r, 2) for r in list(self._reward_window_10)],
            },
            # === WIN / LOSS ===
            "win_loss": {
                "total_wins": self.total_wins,
                "total_losses": self.total_losses,
                "global_win_rate": f"{global_winrate:.1f}%",
                "recent_win_rate_20": f"{winrate_20:.1f}%",
            },
            # === COMBAT ===
            "combat": {
                "total_kills": self.total_kills,
                "total_deaths": self.total_deaths,
                "global_kda_per_match": round(global_kda, 2),
                "avg_kills_last_10": round(avg_kills_10, 2),
                "best_kill_streak": self.best_kill_streak,
                "total_damage_dealt": int(self.total_damage_dealt),
                "total_damage_taken": int(self.total_damage_taken),
                "total_overextend_events": int(self.total_overextend_events),
                "total_low_hp_engage_events": int(self.total_low_hp_engage_events),
                "total_rl_move_overrides": int(self.total_rl_move_overrides),
                "total_rl_move_override_blocks": int(self.total_rl_move_override_blocks),
                "total_rl_override_block_attack_window": int(self.total_rl_override_block_attack_window),
                "total_rl_override_block_pattern_pressure": int(self.total_rl_override_block_pattern_pressure),
                "total_rl_override_critical_applied": int(self.total_rl_override_critical_applied),
                "total_peek_active_frames": int(self.total_peek_active_frames),
                "total_water_pressure_frames": int(self.total_water_pressure_frames),
                "total_enemy_pattern_pressure_frames": int(self.total_enemy_pattern_pressure_frames),
                "total_enemy_attack_soon_frames": int(self.total_enemy_attack_soon_frames),
                "total_kpi_reward_adjustment": round(float(self.total_kpi_reward_adjustment), 4),
            },
            # === VERIFICATION (Survival-First short test) ===
            "verification": {
                "recent_kd_20": round(recent_kd, 3),
                "death_rate_20": round(death_rate_20, 3),
                "damage_trade_ratio_20": round(avg_damage_trade_20, 3),
                "overextend_events_per_match_20": round(overextend_rate_20, 3),
                "low_hp_engages_per_match_20": round(low_hp_engage_rate_20, 3),
                "rl_move_override_applied_per_match_20": round(rl_override_apply_rate_20, 3),
                "rl_move_override_blocked_per_match_20": round(rl_override_block_rate_20, 3),
                "rl_override_block_attack_window_per_match_20": round(rl_override_block_attack_window_rate_20, 3),
                "rl_override_block_pattern_pressure_per_match_20": round(rl_override_block_pattern_pressure_rate_20, 3),
                "rl_override_critical_applied_per_match_20": round(rl_override_critical_applied_rate_20, 3),
                "peek_active_frames_per_match_20": round(peek_active_rate_20, 3),
                "water_pressure_frames_per_match_20": round(water_pressure_rate_20, 3),
                "enemy_pattern_pressure_frames_per_match_20": round(enemy_pattern_pressure_rate_20, 3),
                "enemy_attack_soon_frames_per_match_20": round(enemy_attack_soon_rate_20, 3),
                "kpi_reward_adjustment_per_match_20": round(kpi_reward_adjustment_20, 4),
                "kpi_profile_distribution_20": kpi_profile_distribution_20,
                "one_k_one_d_episode_negative": one_k_one_d_negative,
                "one_k_one_d_samples": len(one_k_one_d_eps),
                "equal_damage_trade_negative": equal_trade_negative,
                "equal_trade_samples": len(equal_trade_eps),
                "value_loss_stable": value_loss_stable,
                "value_loss_spike_count_20": value_spike_count,
                "passive_hide_risk": passive_hide_risk,
            },
            # === REWARD SIGNAL USAGE ===
            "reward_usage": {
                "top_usage_rates": [
                    {"signal": k, "rate": round(v, 6)} for k, v in top_reward_usage
                ],
                "disabled_signals": self._last_disabled_reward_signals,
            },
            # === PER-BRAWLER ===
            "brawler_stats": brawler_summary,
            # === TRAINING LOSSES (last 20) ===
            "loss_history": self.loss_history[-20:],
            # === RECENT MATCHES (last 20) ===
            "recent_matches": self.episode_history[-20:],
        }

        stats_path = os.path.join(save_dir, "training_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"[RL] Model saved to {save_dir}")

    def load(self, path: Optional[str] = None) -> bool:
        """Load model weights and extended stats. Returns True if successful.

        Handles migration from old single-head (policy: 128→108) to new
        multi-head (move_head/attack_head/ability_head) architecture:
          - Shared backbone + value head weights are restored.
          - Old 'policy.*' keys are skipped (incompatible shape).
          - New heads start with fresh random weights → learns from scratch
            but shared features are preserved.
        """
        if not self.is_available:
            return False

        load_dir = path or self.model_dir
        model_path = os.path.join(load_dir, "policy.pt")

        if not os.path.exists(model_path):
            print(f"[RL] No saved model found at {model_path}")
            return False

        try:
            checkpoint = torch.load(model_path, map_location=self.device,
                                     weights_only=True)
            saved_state = checkpoint["model_state_dict"]

            # Check if this is an old single-head model (has 'policy.*' keys)
            has_old_policy = any(k.startswith("policy.") for k in saved_state)
            has_new_heads = any(k.startswith("move_head.") for k in saved_state)

            if has_old_policy and not has_new_heads:
                # === MIGRATION: old → new architecture ===
                # Load only shared + value weights; skip old policy head
                new_state = self.model.state_dict()
                migrated_keys = []
                skipped_keys = []
                for key, value in saved_state.items():
                    if key.startswith("policy."):
                        skipped_keys.append(key)
                        continue  # Skip old single-head policy
                    if key in new_state and new_state[key].shape == value.shape:
                        new_state[key] = value
                        migrated_keys.append(key)
                    else:
                        skipped_keys.append(key)
                self.model.load_state_dict(new_state)
                print(f"[RL] Migrated old model -> multi-head: "
                      f"{len(migrated_keys)} weights restored, "
                      f"{len(skipped_keys)} skipped (old policy head)")
                # Don't load optimizer state — architecture changed
            else:
                # === NORMAL LOAD: same architecture ===
                self.model.load_state_dict(saved_state)
                try:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                except Exception:
                    print("[RL] Optimizer state incompatible, resetting optimizer")

            self.total_updates = checkpoint.get("total_updates", 0)
            self.total_episodes = checkpoint.get("total_episodes", 0)
            self.model.eval()

            # Try to restore extended stats from training_stats.json
            stats_path = os.path.join(load_dir, "training_stats.json")
            if os.path.exists(stats_path):
                try:
                    with open(stats_path, "r") as f:
                        stats = json.load(f)

                    # Detect old flat format vs new structured format
                    if "overview" in stats:
                        # === NEW STRUCTURED FORMAT ===
                        # Restore win/loss
                        wl = stats.get("win_loss", {})
                        self.total_wins = wl.get("total_wins", 0)
                        self.total_losses = wl.get("total_losses", 0)

                        # Restore combat totals
                        combat = stats.get("combat", {})
                        self.total_kills = combat.get("total_kills", 0)
                        self.total_deaths = combat.get("total_deaths", 0)
                        self.total_damage_dealt = combat.get("total_damage_dealt", 0.0)
                        self.total_damage_taken = combat.get("total_damage_taken", 0.0)
                        self.total_overextend_events = combat.get("total_overextend_events", 0)
                        self.total_low_hp_engage_events = combat.get("total_low_hp_engage_events", 0)
                        self.total_rl_move_overrides = combat.get("total_rl_move_overrides", 0)
                        self.total_rl_move_override_blocks = combat.get("total_rl_move_override_blocks", 0)
                        self.total_rl_override_block_attack_window = combat.get("total_rl_override_block_attack_window", 0)
                        self.total_rl_override_block_pattern_pressure = combat.get("total_rl_override_block_pattern_pressure", 0)
                        self.total_rl_override_critical_applied = combat.get("total_rl_override_critical_applied", 0)
                        self.total_peek_active_frames = combat.get("total_peek_active_frames", 0)
                        self.total_water_pressure_frames = combat.get("total_water_pressure_frames", 0)
                        self.total_enemy_pattern_pressure_frames = combat.get("total_enemy_pattern_pressure_frames", 0)
                        self.total_enemy_attack_soon_frames = combat.get("total_enemy_attack_soon_frames", 0)
                        self.total_kpi_reward_adjustment = combat.get("total_kpi_reward_adjustment", 0.0)
                        self.best_kill_streak = combat.get("best_kill_streak", 0)

                        # Restore reward extremes
                        rewards = stats.get("rewards", {})
                        self.best_reward = rewards.get("best_reward", -999999.0)
                        self.worst_reward = rewards.get("worst_reward", 999999.0)
                        # Restore episode history FIRST (before windows)
                        self.episode_history = stats.get("recent_matches", [])
                        self.loss_history = stats.get("loss_history", [])

                        # Restore rolling windows from episode history (more data than last_10)
                        for ep in self.episode_history:
                            r = ep.get("reward", 0.0)
                            self._reward_window_50.append(r)
                            self.avg_reward_window.append(r)
                            if ep.get("won"):
                                self._winrate_window_20.append(1)
                            else:
                                self._winrate_window_20.append(0)
                            self._kill_window_10.append(ep.get("kills", 0))
                        # Ensure last 10 is accurate
                        self._reward_window_10.clear()
                        for ep in self.episode_history[-10:]:
                            self._reward_window_10.append(ep.get("reward", 0.0))

                        # Restore cumulative training hours
                        overview = stats.get("overview", {})
                        self._cumulative_training_hours = overview.get("training_hours", 0.0)

                        # Restore brawler stats
                        for bname, bdata in stats.get("brawler_stats", {}).items():
                            self.brawler_stats[bname] = {
                                "matches": bdata.get("matches", 0),
                                "wins": bdata.get("wins", 0),
                                "losses": bdata.get("losses", 0),
                                "kills": bdata.get("kills", 0),
                                "deaths": bdata.get("deaths", 0),
                                "total_reward": bdata.get("avg_reward", 0) * max(1, bdata.get("matches", 1)),
                                "damage_dealt": bdata.get("avg_damage", 0) * max(1, bdata.get("matches", 1)),
                            }

                        print(f"[RL] Extended stats restored: W{self.total_wins}/L{self.total_losses} "
                              f"K{self.total_kills}/D{self.total_deaths}")
                    else:
                        # === OLD FLAT FORMAT (backwards compat) ===
                        # {"total_updates": N, "total_episodes": N, "avg_reward": X, "last_10_rewards": [...]}
                        for r in stats.get("last_10_rewards", []):
                            self._reward_window_10.append(r)
                            self._reward_window_50.append(r)
                            self.avg_reward_window.append(r)
                        print(f"[RL] Migrated old-format stats (avg_reward={stats.get('avg_reward', 0):.2f})")
                except Exception as e:
                    print(f"[RL] Could not restore extended stats: {e}")

            print(f"[RL] Model loaded from {model_path} (updates={self.total_updates})")
            return True
        except Exception as e:
            print(f"[RL] Failed to load model: {e}")
            return False

    def get_stats_string(self) -> str:
        """Human-readable training stats for debug overlay."""
        avg = np.mean(list(self.avg_reward_window)) if self.avg_reward_window else 0
        tracked = self.total_wins + self.total_losses
        wr = (self.total_wins / max(1, tracked)) * 100
        kda = f"{self.total_kills}/{self.total_deaths}"
        return (f"RL: ep={self.total_episodes} upd={self.total_updates} "
                f"avg_r={avg:.1f} WR={wr:.0f}% K/D={kda} buf={len(self.buffer)}")
