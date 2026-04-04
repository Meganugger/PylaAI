# behavior tree nodes: selector, sequence, conditions, actions, etc.

from __future__ import annotations

import time
import random
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple


# status Enum
class Status(Enum):
    """Result of a behavior tree node tick."""
    SUCCESS = auto()
    FAILURE = auto()
    RUNNING = auto()


# blackboard
class Blackboard:
    """Shared data store for all behavior tree nodes.

    Structured data accessible by all nodes for reading game state
    and writing decisions. Replaces the many self.* variables scattered
    throughout the old Movement/Play classes.
    """

    def __init__(self):
        self.data: Dict[str, Any] = {
            # === PLAYER STATE ===
            "player": {
                "pos": (0, 0),           # (cx, cy) center position
                "bbox": None,            # [x1, y1, x2, y2]
                "hp": 100,               # 0-100 percentage
                "ammo": 3,               # Current ammo count
                "max_ammo": 3,
                "has_super": False,
                "has_gadget": False,
                "has_hypercharge": False,
                "shield_active": False,   # Respawn invincibility
                "is_dead": False,
                "in_storm": False,
                "last_damage_time": 0.0,
                "is_regenerating": False,
                "velocity": (0, 0),       # Estimated movement velocity
            },

            # === ENEMIES ===
            "enemies": [],  # List of enemy dicts:
            # {
            #   "pos": (cx, cy),
            #   "bbox": [x1, y1, x2, y2],
            #   "hp": 0-100 or -1,
            #   "distance": float,
            #   "velocity": (vx, vy),
            #   "brawler": str or None,     # Identified brawler name
            #   "cooldown_est": float,       # Estimated seconds until next attack
            #   "threat_score": float,       # 0-100 composite threat rating
            #   "is_hittable": bool,         # Line of sight clear
            #   "last_seen": float,          # timestamp
            #   "reload_speed": float,       # from brawlers_info if identified
            #   "attack_range": float,       # from brawlers_info if identified
            # }

            # === TEAMMATES ===
            "teammates": [],  # List of teammate dicts:
            # {
            #   "pos": (cx, cy),
            #   "bbox": [x1, y1, x2, y2],
            #   "hp": 0-100 or -1,
            #   "distance": float,
            # }

            # === MAP / SPATIAL ===
            "map": {
                "walls": [],             # List of wall bboxes
                "bushes": [],            # List of bush bboxes
                "grid": None,            # SpatialMemory occupancy grid
                "danger_zones": [],      # List of (x, y, radius, score) danger areas
                "choke_points": [],      # List of (cx, cy, width, angle)
                "destroyed_walls": [],   # List of (x1, y1, x2, y2)
                "storm_center": (960, 540),
                "storm_radius": 9999,
                "gas_active": False,
            },

            # === PROJECTILES ===
            "projectiles": [],  # List of projectile dicts:
            # {
            #   "pos": (x, y),
            #   "velocity": (vx, vy),
            #   "radius": float,
            #   "threat_level": float,    # 0-1 how dangerous
            #   "frames_to_impact": int,  # estimated frames until hits player
            # }

            # === MATCH STATE ===
            "match": {
                "phase": "early",        # "early" / "mid" / "late"
                "time_elapsed": 0.0,
                "our_score": 0,
                "their_score": 0,
                "score_diff": 0,
                "mode": "knockout",
                "is_showdown": False,
                "spawn_side": None,
            },

            # === BRAWLER INFO ===
            "brawler": {
                "name": "",
                "playstyle": "fighter",
                "attack_range": 400,
                "safe_range": 300,
                "super_range": 400,
                "super_type": "damage",
                "reload_speed": 1.4,
                "health": 3200,
                "attack_damage": 1200,
                "projectile_count": 1,
                "movement_speed": 720,
                "hold_attack": 0,
                "ignore_walls_for_attacks": False,
                "ignore_walls_for_supers": False,
            },

            # === DECISION OUTPUT ===
            "decision": {
                "movement": "",          # Direction keys string (e.g., "WA")
                "should_attack": False,
                "attack_type": "auto",   # "auto" / "aimed" / "super" / "gadget" / "hypercharge"
                "aim_target": None,      # (x, y) for aimed attacks
                "target_enemy": None,    # Reference to target enemy dict
                "reason": "",            # Human-readable decision reason
                "combo": None,           # Active combo sequence name
            },

            # === PLAYSTYLE CONFIG (loaded per brawler) ===
            "playstyle_config": {},

            # === AGGRESSION ===
            "aggression": 1.0,           # 0.7=defensive, 1.3=aggressive

            # === TIMING ===
            "current_time": 0.0,
            "dt": 0.0,                   # Delta time since last tick
            "last_tick_time": 0.0,
        }

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any):
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value. Supports dot-path keys: 'player.hp' -> data['player']['hp'].
        Falls back to flat key lookup for backward compatibility."""
        if "." in key:
            return self.get_nested(*key.split("."), default=default)
        return self.data.get(key, default)

    def set(self, key: str, value: Any):
        """Set a value. Supports dot-path keys: 'player.hp' -> data['player']['hp'] = value.
        Falls back to flat key assignment for plain keys."""
        if "." in key:
            self.set_nested(*key.split("."), value)
        else:
            self.data[key] = value

    def get_nested(self, *keys: str, default: Any = None) -> Any:
        """Get a nested value safely. E.g., bb.get_nested('player', 'hp')"""
        d = self.data
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k, default)
            else:
                return default
        return d

    def set_nested(self, *keys_and_value):
        """Set a nested value. Last argument is the value.
        E.g., bb.set_nested('player', 'hp', 50)"""
        keys = keys_and_value[:-1]
        value = keys_and_value[-1]
        d = self.data
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value


# base Node
class Node(ABC):
    """Abstract base class for all behavior tree nodes."""

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
        self._status = Status.FAILURE
        self._tick_count = 0

    @abstractmethod
    def tick(self, bb: Blackboard) -> Status:
        """Execute this node's logic and return a Status."""
        ...

    def reset(self):
        """Reset node state (called when tree re-enters this branch)."""
        self._status = Status.FAILURE
        self._tick_count = 0

    @property
    def status(self) -> Status:
        return self._status

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r})"


# composite Nodes
class Composite(Node):
    """Base class for nodes with children."""

    def __init__(self, name: str, children: List[Node]):
        super().__init__(name)
        self.children = children

    def reset(self):
        super().reset()
        for child in self.children:
            child.reset()


class Sequence(Composite):
    """Executes children in order. Returns FAILURE on first failure.
    Returns RUNNING if a child returns RUNNING (resumes from that child next tick).
    Returns SUCCESS only if ALL children succeed.

    Think of this as an AND gate.
    """

    def __init__(self, name: str, children: List[Node]):
        super().__init__(name, children)
        self._running_child = 0

    def tick(self, bb: Blackboard) -> Status:
        self._tick_count += 1
        for i in range(self._running_child, len(self.children)):
            status = self.children[i].tick(bb)
            if status == Status.RUNNING:
                self._running_child = i
                self._status = Status.RUNNING
                return Status.RUNNING
            elif status == Status.FAILURE:
                self._running_child = 0
                self._status = Status.FAILURE
                return Status.FAILURE
        self._running_child = 0
        self._status = Status.SUCCESS
        return Status.SUCCESS

    def reset(self):
        super().reset()
        self._running_child = 0


class Selector(Composite):
    """Executes children in order. Returns SUCCESS on first success.
    Returns RUNNING if a child returns RUNNING (resumes from that child next tick).
    Returns FAILURE only if ALL children fail.

    Think of this as an OR gate (fallback).
    """

    def __init__(self, name: str, children: List[Node]):
        super().__init__(name, children)
        self._running_child = 0

    def tick(self, bb: Blackboard) -> Status:
        self._tick_count += 1
        for i in range(self._running_child, len(self.children)):
            status = self.children[i].tick(bb)
            if status == Status.RUNNING:
                self._running_child = i
                self._status = Status.RUNNING
                return Status.RUNNING
            elif status == Status.SUCCESS:
                self._running_child = 0
                self._status = Status.SUCCESS
                return Status.SUCCESS
        self._running_child = 0
        self._status = Status.FAILURE
        return Status.FAILURE

    def reset(self):
        super().reset()
        self._running_child = 0


class Parallel(Composite):
    """Ticks all children simultaneously.
    Success policy: 'all' = all must succeed, 'one' = one success is enough.
    Failure policy: 'all' = all must fail, 'one' = one failure is enough.
    """

    def __init__(self, name: str, children: List[Node],
                 success_policy: str = "all", failure_policy: str = "one"):
        super().__init__(name, children)
        self.success_policy = success_policy
        self.failure_policy = failure_policy

    def tick(self, bb: Blackboard) -> Status:
        self._tick_count += 1
        successes = 0
        failures = 0

        for child in self.children:
            status = child.tick(bb)
            if status == Status.SUCCESS:
                successes += 1
            elif status == Status.FAILURE:
                failures += 1

        # Check failure first
        if self.failure_policy == "one" and failures > 0:
            self._status = Status.FAILURE
            return Status.FAILURE
        if self.failure_policy == "all" and failures == len(self.children):
            self._status = Status.FAILURE
            return Status.FAILURE

        # Check success
        if self.success_policy == "all" and successes == len(self.children):
            self._status = Status.SUCCESS
            return Status.SUCCESS
        if self.success_policy == "one" and successes > 0:
            self._status = Status.SUCCESS
            return Status.SUCCESS

        self._status = Status.RUNNING
        return Status.RUNNING


# decorator Nodes
class Decorator(Node):
    """Base class for nodes that modify a single child's behavior."""

    def __init__(self, name: str, child: Node):
        super().__init__(name)
        self.child = child

    def reset(self):
        super().reset()
        self.child.reset()


class Inverter(Decorator):
    """Inverts child result: SUCCESS↔FAILURE, RUNNING unchanged."""

    def tick(self, bb: Blackboard) -> Status:
        status = self.child.tick(bb)
        if status == Status.SUCCESS:
            self._status = Status.FAILURE
            return Status.FAILURE
        elif status == Status.FAILURE:
            self._status = Status.SUCCESS
            return Status.SUCCESS
        self._status = Status.RUNNING
        return Status.RUNNING


class ForceSuccess(Decorator):
    """Always returns SUCCESS regardless of child result."""

    def tick(self, bb: Blackboard) -> Status:
        self.child.tick(bb)
        self._status = Status.SUCCESS
        return Status.SUCCESS


class ForceFailure(Decorator):
    """Always returns FAILURE regardless of child result."""

    def tick(self, bb: Blackboard) -> Status:
        self.child.tick(bb)
        self._status = Status.FAILURE
        return Status.FAILURE


class CooldownGuard(Decorator):
    """Only allows child to execute once every `cooldown` seconds.
    Returns FAILURE during cooldown period."""

    def __init__(self, name: str, child: Node, cooldown: float):
        super().__init__(name, child)
        self.cooldown = cooldown
        self._last_execution = 0.0

    def tick(self, bb: Blackboard) -> Status:
        now = bb["current_time"] or time.time()
        if now - self._last_execution < self.cooldown:
            self._status = Status.FAILURE
            return Status.FAILURE
        status = self.child.tick(bb)
        if status != Status.RUNNING:
            self._last_execution = now
        self._status = status
        return status

    def reset(self):
        super().reset()
        self._last_execution = 0.0


class Repeater(Decorator):
    """Repeats child N times. Returns SUCCESS after N successes.
    If child fails, returns FAILURE immediately unless fail_ok=True."""

    def __init__(self, name: str, child: Node, count: int, fail_ok: bool = False):
        super().__init__(name, child)
        self.count = count
        self.fail_ok = fail_ok
        self._current = 0

    def tick(self, bb: Blackboard) -> Status:
        while self._current < self.count:
            status = self.child.tick(bb)
            if status == Status.RUNNING:
                self._status = Status.RUNNING
                return Status.RUNNING
            elif status == Status.FAILURE and not self.fail_ok:
                self._current = 0
                self._status = Status.FAILURE
                return Status.FAILURE
            self._current += 1
        self._current = 0
        self._status = Status.SUCCESS
        return Status.SUCCESS

    def reset(self):
        super().reset()
        self._current = 0


class ConditionalGuard(Decorator):
    """Only ticks child if condition function returns True.
    Otherwise returns FAILURE without ticking child."""

    def __init__(self, name: str, child: Node, condition: Callable[[Blackboard], bool]):
        super().__init__(name, child)
        self.condition = condition

    def tick(self, bb: Blackboard) -> Status:
        if self.condition(bb):
            status = self.child.tick(bb)
            self._status = status
            return status
        self._status = Status.FAILURE
        return Status.FAILURE


class RandomChance(Decorator):
    """Ticks child with a given probability (0.0-1.0).
    Returns FAILURE if random check fails."""

    def __init__(self, name: str, child: Node, probability: float):
        super().__init__(name, child)
        self.probability = max(0.0, min(1.0, probability))

    def tick(self, bb: Blackboard) -> Status:
        if random.random() < self.probability:
            status = self.child.tick(bb)
            self._status = status
            return status
        self._status = Status.FAILURE
        return Status.FAILURE


class TimeLimit(Decorator):
    """Limits child execution time. Returns FAILURE if time exceeded."""

    def __init__(self, name: str, child: Node, max_seconds: float):
        super().__init__(name, child)
        self.max_seconds = max_seconds
        self._start_time = 0.0
        self._started = False

    def tick(self, bb: Blackboard) -> Status:
        now = bb["current_time"] or time.time()
        if not self._started:
            self._start_time = now
            self._started = True

        if now - self._start_time > self.max_seconds:
            self._started = False
            self.child.reset()
            self._status = Status.FAILURE
            return Status.FAILURE

        status = self.child.tick(bb)
        if status != Status.RUNNING:
            self._started = False
        self._status = status
        return status

    def reset(self):
        super().reset()
        self._started = False
        self._start_time = 0.0


# leaf Nodes
class Condition(Node):
    """Leaf node that checks a condition against the blackboard.
    Returns SUCCESS if condition is True, FAILURE otherwise.
    Conditions should be pure (no side effects).
    """

    def __init__(self, name: str, check: Callable[[Blackboard], bool]):
        super().__init__(name)
        self.check = check

    def tick(self, bb: Blackboard) -> Status:
        self._tick_count += 1
        result = self.check(bb)
        self._status = Status.SUCCESS if result else Status.FAILURE
        return self._status


class Action(Node):
    """Leaf node that executes an action function.
    The action receives the blackboard and should return a Status.
    If it returns None, SUCCESS is assumed.
    """

    def __init__(self, name: str, action: Callable[[Blackboard], Optional[Status]]):
        super().__init__(name)
        self.action = action

    def tick(self, bb: Blackboard) -> Status:
        self._tick_count += 1
        result = self.action(bb)
        if result is None:
            result = Status.SUCCESS
        self._status = result
        return self._status


class SetBlackboard(Node):
    """Leaf node that sets a value on the blackboard. Always returns SUCCESS."""

    def __init__(self, name: str, key_path: Tuple[str, ...], value_fn: Callable[[Blackboard], Any]):
        super().__init__(name)
        self.key_path = key_path
        self.value_fn = value_fn

    def tick(self, bb: Blackboard) -> Status:
        value = self.value_fn(bb)
        bb.set_nested(*self.key_path, value)
        self._status = Status.SUCCESS
        return Status.SUCCESS


class WaitSeconds(Node):
    """Leaf that returns RUNNING for a specified duration, then SUCCESS."""

    def __init__(self, name: str, duration: float):
        super().__init__(name)
        self.duration = duration
        self._start_time = 0.0
        self._started = False

    def tick(self, bb: Blackboard) -> Status:
        now = bb["current_time"] or time.time()
        if not self._started:
            self._start_time = now
            self._started = True

        if now - self._start_time >= self.duration:
            self._started = False
            self._status = Status.SUCCESS
            return Status.SUCCESS

        self._status = Status.RUNNING
        return Status.RUNNING

    def reset(self):
        super().reset()
        self._started = False
        self._start_time = 0.0


class Lambda(Node):
    """Quick inline node - wraps any callable that returns Status or bool."""

    def __init__(self, name: str, fn: Callable[[Blackboard], Any]):
        super().__init__(name)
        self.fn = fn

    def tick(self, bb: Blackboard) -> Status:
        result = self.fn(bb)
        if isinstance(result, Status):
            self._status = result
        elif isinstance(result, bool):
            self._status = Status.SUCCESS if result else Status.FAILURE
        else:
            self._status = Status.SUCCESS
        return self._status


# utility / Weighted Selector
class WeightedSelector(Composite):
    """Scores each child using a weight function and ticks the highest-scoring one.
    Useful for utility-AI-style decisions within the behavior tree.

    weight_fns: List of callables (one per child) returning float scores.
    If a child's weight is ≤ 0, it's skipped.
    """

    def __init__(self, name: str, children: List[Node],
                 weight_fns: List[Callable[[Blackboard], float]]):
        super().__init__(name, children)
        assert len(weight_fns) == len(children), "Must have one weight function per child"
        self.weight_fns = weight_fns

    def tick(self, bb: Blackboard) -> Status:
        self._tick_count += 1
        scores = [(self.weight_fns[i](bb), i) for i in range(len(self.children))]
        scores.sort(key=lambda x: x[0], reverse=True)

        for score, idx in scores:
            if score <= 0:
                continue
            status = self.children[idx].tick(bb)
            if status in (Status.SUCCESS, Status.RUNNING):
                self._status = status
                return status

        self._status = Status.FAILURE
        return Status.FAILURE


# tree Debugging
def tree_to_string(node: Node, indent: int = 0) -> str:
    """Pretty-print the behavior tree structure."""
    prefix = "  " * indent
    status_char = {Status.SUCCESS: "OK", Status.FAILURE: "FAIL", Status.RUNNING: "->"}.get(
        node.status, "?"
    )
    result = f"{prefix}[{status_char}] {node}\n"
    if isinstance(node, Composite):
        for child in node.children:
            result += tree_to_string(child, indent + 1)
    elif isinstance(node, Decorator):
        result += tree_to_string(node.child, indent + 1)
    return result


def get_active_path(node: Node) -> List[str]:
    """Get the path of currently active/running nodes (for overlay display)."""
    path = [node.name]
    if isinstance(node, Composite):
        for child in node.children:
            if child.status in (Status.SUCCESS, Status.RUNNING):
                path.extend(get_active_path(child))
                break
    elif isinstance(node, Decorator):
        if node.child.status in (Status.SUCCESS, Status.RUNNING):
            path.extend(get_active_path(node.child))
    return path
