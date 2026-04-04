# timed ability/attack combos (burst sequences for different brawler types)

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple


# combo Definitions
# Each combo is a list of (action_name, delay_ms_after_previous)
# action_name maps to a function that performs the action

COMBO_DEFINITIONS: Dict[str, Dict[str, List[Tuple[str, int]]]] = {
    # === ASSASSIN COMBOS ===
    "assassin": {
        "super_burst": [
            ("super", 0),           # Super to close gap
            ("attack", 120),        # Immediate attack on arrival
            ("attack", 200),        # Follow-up attack
            ("gadget", 150),        # Gadget for extra damage/escape
        ],
        "full_dump": [
            ("attack", 0),          # First shot
            ("attack", 60),         # Rapid follow-up
            ("attack", 60),         # Third shot
            ("super", 120),         # Super to finish or escape
        ],
        "gadget_engage": [
            ("gadget", 0),          # Gadget to close distance / buff
            ("attack", 100),        # Immediate attack
            ("attack", 80),         # Follow-up
        ],
    },

    # === FIGHTER COMBOS ===
    "fighter": {
        "super_then_attack": [
            ("super", 0),           # Super for damage/zone
            ("attack", 120),        # Follow-up attack
        ],
        "burst_3": [
            ("attack", 0),
            ("attack", 80),
            ("attack", 80),
        ],
        "gadget_burst": [
            ("gadget", 0),          # Gadget buff
            ("attack", 100),
            ("attack", 80),
            ("attack", 80),
        ],
    },

    # === TANK COMBOS ===
    "tank": {
        "charge_burst": [
            ("super", 0),           # Super to charge in (Bull/Darryl/El Primo)
            ("attack", 150),        # Attack on arrival
            ("attack", 60),         # Spam at close range
            ("attack", 60),
        ],
        "point_blank": [
            ("attack", 0),
            ("attack", 50),         # Tanks have fast close-range fire
            ("attack", 50),
            ("gadget", 100),        # Gadget for sustain/damage
        ],
    },

    # === SNIPER COMBOS ===
    "sniper": {
        "double_tap": [
            ("attack", 0),
            ("attack", 300),        # Snipers fire slower but more precise
        ],
        "super_snipe": [
            ("super", 0),           # Some snipers have damage supers
            ("attack", 200),
        ],
    },

    # === THROWER COMBOS ===
    "thrower": {
        "area_deny": [
            ("attack", 0),
            ("attack", 200),        # Spread shots for area denial
            ("attack", 200),
        ],
        "super_area": [
            ("super", 0),           # Area super
            ("attack", 150),
            ("attack", 150),
        ],
        "gadget_area": [
            ("gadget", 0),          # Gadget (often area effect)
            ("attack", 100),
            ("attack", 150),
        ],
    },

    # === SUPPORT COMBOS ===
    "support": {
        "heal_then_fight": [
            ("super", 0),           # Heal super
            ("attack", 200),
        ],
        "burst_2": [
            ("attack", 0),
            ("attack", 100),
        ],
    },
}


# combo Trigger Conditions
def _default_trigger_conditions() -> Dict[str, Callable]:
    """Returns condition checkers for each combo type.
    Each returns True if the combo should be triggered."""
    return {
        # Requires: super ready + enemy in range
        "super_burst": lambda bb: (
            bb["player"]["has_super"]
            and len(bb["enemies"]) > 0
            and bb["enemies"][0].get("distance", 999) < bb["brawler"]["super_range"]
            and bb["enemies"][0].get("is_hittable", False)
        ),

        # Requires: 3 ammo + enemy in attack range
        "full_dump": lambda bb: (
            bb["player"]["ammo"] >= 3
            and len(bb["enemies"]) > 0
            and bb["enemies"][0].get("distance", 999) < bb["brawler"]["attack_range"]
            and bb["enemies"][0].get("is_hittable", False)
        ),

        # Requires: gadget ready + enemy close
        "gadget_engage": lambda bb: (
            bb["player"]["has_gadget"]
            and len(bb["enemies"]) > 0
            and bb["enemies"][0].get("distance", 999) < bb["brawler"]["attack_range"] * 0.8
        ),

        "super_then_attack": lambda bb: (
            bb["player"]["has_super"]
            and len(bb["enemies"]) > 0
            and bb["enemies"][0].get("distance", 999) < bb["brawler"]["super_range"]
        ),

        "burst_3": lambda bb: (
            bb["player"]["ammo"] >= 3
            and len(bb["enemies"]) > 0
            and bb["enemies"][0].get("distance", 999) < bb["brawler"]["attack_range"]
        ),

        "gadget_burst": lambda bb: (
            bb["player"]["has_gadget"]
            and bb["player"]["ammo"] >= 2
            and len(bb["enemies"]) > 0
            and bb["enemies"][0].get("distance", 999) < bb["brawler"]["attack_range"]
        ),

        "charge_burst": lambda bb: (
            bb["player"]["has_super"]
            and len(bb["enemies"]) > 0
            and bb["enemies"][0].get("distance", 999) < bb["brawler"]["super_range"]
            and bb["player"]["hp"] > 40  # Don't charge in with low HP
        ),

        "point_blank": lambda bb: (
            bb["player"]["ammo"] >= 3
            and len(bb["enemies"]) > 0
            and bb["enemies"][0].get("distance", 999) < 150  # Very close
        ),

        "double_tap": lambda bb: (
            bb["player"]["ammo"] >= 2
            and len(bb["enemies"]) > 0
            and bb["enemies"][0].get("distance", 999) < bb["brawler"]["attack_range"]
            and bb["enemies"][0].get("is_hittable", False)
        ),

        "super_snipe": lambda bb: (
            bb["player"]["has_super"]
            and len(bb["enemies"]) > 0
            and bb["enemies"][0].get("distance", 999) < bb["brawler"]["super_range"]
            and bb["enemies"][0].get("is_hittable", False)
        ),

        "area_deny": lambda bb: (
            bb["player"]["ammo"] >= 3
            and len(bb["enemies"]) > 0
            and bb["enemies"][0].get("distance", 999) < bb["brawler"]["attack_range"]
        ),

        "super_area": lambda bb: (
            bb["player"]["has_super"]
            and len(bb["enemies"]) >= 2  # Multiple enemies for area supers
        ),

        "gadget_area": lambda bb: (
            bb["player"]["has_gadget"]
            and len(bb["enemies"]) > 0
        ),

        "heal_then_fight": lambda bb: (
            bb["player"]["has_super"]
            and bb["player"]["hp"] < 60
            and len(bb["teammates"]) > 0
        ),

        "burst_2": lambda bb: (
            bb["player"]["ammo"] >= 2
            and len(bb["enemies"]) > 0
            and bb["enemies"][0].get("distance", 999) < bb["brawler"]["attack_range"]
        ),
    }


# combo Engine
class ComboEngine:
    """Manages and executes combo sequences.

    The engine queues actions with precise timing and executes them
    during the main game loop via tick().
    """

    def __init__(self, window_controller=None):
        self.wc = window_controller  # WindowController for input injection

        # Current combo state
        self._active_combo: Optional[str] = None
        self._combo_sequence: List[Tuple[str, int]] = []
        self._combo_step: int = 0
        self._combo_start_time: float = 0.0
        self._next_action_time: float = 0.0
        self._combo_playstyle: str = "fighter"

        # Available combos for current brawler
        self._available_combos: Dict[str, List[Tuple[str, int]]] = {}
        self._trigger_conditions = _default_trigger_conditions()

        # Combo cooldown (don't spam combos)
        self._last_combo_end: float = 0.0
        self._combo_cooldown: float = 2.0  # seconds between combos
        self._combo_priority: List[str] = []  # Ordered by priority

        # Action execution callbacks (set by the caller)
        self._action_map: Dict[str, Callable] = {}

        # Stats
        self.combos_executed: int = 0
        self.combos_interrupted: int = 0

    def load_combos(self, playstyle: str):
        """Load combo definitions for a specific playstyle."""
        self._combo_playstyle = playstyle
        self._available_combos = COMBO_DEFINITIONS.get(playstyle, {})

        # Set priority order (super combos > gadget combos > basic combos)
        super_combos = [k for k in self._available_combos if "super" in k or "charge" in k]
        gadget_combos = [k for k in self._available_combos if "gadget" in k]
        basic_combos = [k for k in self._available_combos if k not in super_combos and k not in gadget_combos]
        self._combo_priority = super_combos + gadget_combos + basic_combos

    def set_action_handlers(self, handlers: Dict[str, Callable]):
        """Set the functions that execute each action type.

        Expected handlers:
            "attack": fn() - fire normal attack
            "super": fn() - use super ability
            "gadget": fn() - use gadget
            "hypercharge": fn() - use hypercharge
            "aimed_attack": fn(target_pos) - aimed attack at position
            "aimed_super": fn(target_pos) - aimed super at position
        """
        self._action_map = handlers

    def should_trigger(self, blackboard) -> Optional[str]:
        """Check if any combo should be triggered based on current game state.

        Returns: combo name if a combo should start, None otherwise.
        """
        now = time.time()
        if self.is_active:
            return None

        if now - self._last_combo_end < self._combo_cooldown:
            return None

        for combo_name in self._combo_priority:
            if combo_name not in self._available_combos:
                continue
            condition = self._trigger_conditions.get(combo_name)
            if condition and condition(blackboard):
                return combo_name

        return None

    def start(self, combo_name: str) -> bool:
        """Start executing a combo sequence.

        Returns: True if combo started, False if invalid or already active.
        """
        if combo_name not in self._available_combos:
            return False

        if self.is_active:
            self.interrupt()

        self._active_combo = combo_name
        self._combo_sequence = self._available_combos[combo_name]
        self._combo_step = 0
        self._combo_start_time = time.time()
        self._next_action_time = self._combo_start_time

        # Execute the first action immediately if delay is 0
        if self._combo_sequence and self._combo_sequence[0][1] == 0:
            self._execute_current_step()

        return True

    def tick(self, current_time: float) -> Optional[str]:
        """Process the combo queue. Call every frame.

        Returns: The action executed this tick, or None.
        """
        if not self.is_active:
            return None

        if self._combo_step >= len(self._combo_sequence):
            self._finish_combo()
            return None

        if current_time >= self._next_action_time:
            action = self._execute_current_step()
            return action

        return None

    def _execute_current_step(self) -> Optional[str]:
        """Execute the current combo step and advance to next."""
        if self._combo_step >= len(self._combo_sequence):
            return None

        action_name, delay_ms = self._combo_sequence[self._combo_step]

        # Execute the action
        handler = self._action_map.get(action_name)
        if handler:
            try:
                handler()
            except Exception as e:
                print(f"[COMBO] Error executing {action_name}: {e}")
                self.interrupt()
                return None

        self._combo_step += 1

        # Schedule next action
        if self._combo_step < len(self._combo_sequence):
            next_delay_ms = self._combo_sequence[self._combo_step][1]
            self._next_action_time = time.time() + next_delay_ms / 1000.0
        else:
            # Combo finished
            self._finish_combo()

        return action_name

    def _finish_combo(self):
        """Complete the current combo."""
        self._active_combo = None
        self._combo_sequence = []
        self._combo_step = 0
        self._last_combo_end = time.time()
        self.combos_executed += 1

    def interrupt(self):
        """Interrupt the current combo (e.g., because we need to dodge)."""
        if self.is_active:
            self._active_combo = None
            self._combo_sequence = []
            self._combo_step = 0
            self.combos_interrupted += 1

    @property
    def is_active(self) -> bool:
        """Whether a combo is currently being executed."""
        return self._active_combo is not None

    @property
    def active_combo_name(self) -> Optional[str]:
        return self._active_combo

    @property
    def progress(self) -> float:
        """Combo progress 0.0-1.0."""
        if not self.is_active or not self._combo_sequence:
            return 0.0
        return self._combo_step / len(self._combo_sequence)

    @property
    def remaining_steps(self) -> int:
        if not self.is_active:
            return 0
        return len(self._combo_sequence) - self._combo_step

    def get_status_string(self) -> str:
        """Human-readable combo status for debug overlay."""
        if not self.is_active:
            return "No combo"
        step = self._combo_step
        total = len(self._combo_sequence)
        return f"{self._active_combo} [{step}/{total}]"

    def reset(self):
        """Reset combo state for new match."""
        self._active_combo = None
        self._combo_sequence = []
        self._combo_step = 0
        self._last_combo_end = 0.0
        self.combos_executed = 0
        self.combos_interrupted = 0

    def try_start(self, blackboard, playstyle: str) -> bool:
        """Convenience: load combos for playstyle, check triggers, start if possible.

        Returns True if a combo was started.
        """
        if self.is_active:
            return False
        if self._combo_playstyle != playstyle:
            self.load_combos(playstyle)
        combo_name = self.should_trigger(blackboard)
        if combo_name:
            return self.start(combo_name)
        return False
