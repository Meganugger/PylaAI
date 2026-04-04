# builds + ticks the behavior tree each frame, reads back movement/attack decisions

from __future__ import annotations

import math
import time
import random
from typing import Optional, Tuple, Dict, Any

from behavior_tree import (
    Blackboard, Status,
    Selector, Sequence, Parallel,
    Condition, Action, Lambda,
    CooldownGuard, ConditionalGuard, RandomChance,
    SetBlackboard, WaitSeconds, ForceSuccess,
    tree_to_string, get_active_path,
)

# module-level pathfinder reference (set during BTCombat.tick)
# This avoids modifying all 17 call sites of _compute_movement_toward/away.
_active_pathfinder = None  # pathfinder.PathPlanner or None
_active_spatial_memory = None  # spatial_memory.SpatialMemory or None
_active_storm_center = None  # (x, y) safe zone center for flee logic
# TRAP print throttling - initialize to 0
_last_trap_print_time = 0.0
_last_concavity_print_time = 0.0


def _lookup_enemy_hp(play_instance, enemy_pos) -> int:
    """Look up HP for a specific enemy from the per-enemy HP map.

    Matches by proximity to the enemy's center position.
    Falls back to the global enemy_hp_percent for the closest enemy.
    """
    per_hp = getattr(play_instance, '_per_enemy_hp', {})
    if not per_hp:
        # No per-enemy data available - fall back to closest enemy HP
        return getattr(play_instance, 'enemy_hp_percent', -1)

    ecx, ecy = enemy_pos
    best_key = None
    best_dist = 999999
    for key, (hp, conf, ts) in per_hp.items():
        try:
            kx, ky = [float(v) for v in key.split(",")]
        except (ValueError, AttributeError):
            continue
        d = abs(kx - ecx) + abs(ky - ecy)  # Manhattan distance
        if d < best_dist:
            best_dist = d
            best_key = key

    if best_key and best_dist < 80:  # Within ~80px match threshold
        return per_hp[best_key][0]

    # No close match - fall back
    return getattr(play_instance, 'enemy_hp_percent', -1)


# --- bLACKBOARD POPULATION ---

def populate_blackboard(bb: Blackboard, play_instance, data: dict,
                         frame, brawler: str):
    """Fill the blackboard from current play.py state + detection data.

    """
    p = play_instance  # shorthand
    now = time.time()

    # player
    player_bbox = data.get('player', [[0, 0, 0, 0]])[0] if data.get('player') else [0, 0, 0, 0]
    if p and data.get('player'):
        try:
            player_pos = p.get_player_pos(player_bbox)
        except Exception:
            player_pos = ((player_bbox[0] + player_bbox[2]) / 2,
                          (player_bbox[1] + player_bbox[3]) / 2)
    elif data.get('mainData'):
        md = data['mainData'][0]
        player_pos = ((md[0] + md[2]) / 2, (md[1] + md[3]) / 2)
    else:
        player_pos = (0, 0)

    bb.set("player.pos", player_pos)
    bb.set("player.hp", getattr(p, 'player_hp_percent', 100))
    bb.set("player.hp_confidence", getattr(p, '_hp_confidence_player', 1.0))
    bb.set("player.hp_state", getattr(p, '_hp_state_player', 'healthy'))
    bb.set("player.hp_age", getattr(p, '_hp_data_age_player', 0.0))
    bb.set("player.ammo", getattr(p, '_ammo', 3))
    bb.set("player.max_ammo", getattr(p, '_max_ammo', 3))
    bb.set("player.super_ready", getattr(p, 'is_super_ready', False))
    bb.set("player.gadget_ready", getattr(p, 'is_gadget_ready', False))
    bb.set("player.hypercharge_ready", getattr(p, 'is_hypercharge_ready', False))
    bb.set("player.is_dead", getattr(p, '_is_dead', False))
    bb.set("player.respawn_shield", getattr(p, '_respawn_shield_active', False))
    bb.set("player.bbox", player_bbox)

    # brawler info
    brawler_info = getattr(p, 'brawlers_info', {}).get(brawler, {}) if p else {}
    playstyle = brawler_info.get('playstyle', 'fighter')
    bb.set("brawler.name", brawler or "")
    bb.set("brawler.playstyle", playstyle)
    bb.set("brawler.attack_range", brawler_info.get('attack_range', 300))
    bb.set("brawler.safe_range", brawler_info.get('safe_range', 200))
    bb.set("brawler.super_range", brawler_info.get('super_range', 300))
    bb.set("brawler.attack_damage", brawler_info.get('attack_damage', 1200))
    bb.set("brawler.health", brawler_info.get('health', 3200))
    bb.set("brawler.reload_speed", brawler_info.get('reload_speed', 1.4))
    bb.set("brawler.movement_speed", brawler_info.get('movement_speed', 720))

    # Scaled ranges
    try:
        safe_r, atk_r, super_r = p.get_brawler_range(brawler) if p else (200, 300, 300)
    except Exception:
        safe_r, atk_r, super_r = 200, 300, 300
    bb.set("brawler.safe_range_scaled", safe_r)
    bb.set("brawler.attack_range_scaled", atk_r)
    bb.set("brawler.super_range_scaled", super_r)

    # enemies
    enemies_raw = data.get('enemy', [])
    enemies = enemies_raw if enemies_raw else []
    walls = data.get('wall', []) or []
    enemy_list = []
    # Throwers lob projectiles OVER walls - wall LOS is irrelevant for them!
    _playstyle = bb.get("brawler.playstyle", "fighter")
    _skip_wall_los = _playstyle in ("thrower",)
    # Very close threshold: allow LOS bypass when enemy is very close,
    # because wall boxes often overlap brawler sprites at close combat.
    _melee_skip_threshold = 95
    for en in enemies:
        if p:
            try:
                epos = p.get_enemy_pos(en)
                edist = p.get_distance(epos, player_pos)
                # Throwers always hittable (projectiles arc over walls)
                if _skip_wall_los:
                    hittable = True
                # Only check wall blocking if walls are detected
                elif walls:
                    # Only skip LOS at point-blank melee range (<60px) where
                    # wall boxes may overlap the player sprite itself.
                    if edist < _melee_skip_threshold:
                        hittable = True
                    else:
                        hittable = p.is_enemy_hittable(player_pos, epos, walls, "attack")
                else:
                    hittable = True  # No walls = always hittable
            except Exception:
                epos = ((en[0] + en[2]) / 2, (en[1] + en[3]) / 2)
                edist = math.sqrt((epos[0] - player_pos[0])**2 + (epos[1] - player_pos[1])**2)
                hittable = True
        else:
            epos = ((en[0] + en[2]) / 2, (en[1] + en[3]) / 2)
            edist = math.sqrt((epos[0] - player_pos[0])**2 + (epos[1] - player_pos[1])**2)
            hittable = True
        enemy_list.append({
            "bbox": en,
            "pos": epos,
            "distance": edist,
            "hittable": hittable,
            "hp": _lookup_enemy_hp(p, epos) if p else -1,
        })

    # Sort by distance (closest first)
    enemy_list.sort(key=lambda e: e["distance"])
    bb["enemies"] = enemy_list
    bb["enemies_count"] = len(enemy_list)
    bb["enemies_closest"] = enemy_list[0] if enemy_list else None

    # enemy velocity (for predictive aiming)
    bb.set("enemy.velocity", getattr(p, '_enemy_velocity_smooth', (0, 0)) if p else (0, 0))
    bb.set("enemy.speed", getattr(p, '_enemy_speed_magnitude', 0.0) if p else 0.0)
    bb.set("enemy.move_direction", getattr(p, '_enemy_move_direction', 'none') if p else 'none')
    bb.set("enemy.pattern_pressure", 0.0)
    bb.set("enemy.strafe_ratio", 0.0)
    bb.set("enemy.approach_ratio", 0.0)
    bb.set("enemy.predicted_attack_in", 999.0)
    bb.set("enemy.safety_window", 0.0)
    bb.set("enemy.predicted_attack_soon", 0.0)
    bb.set("player.peek_phase", getattr(p, '_peek_phase', 'idle') if p else 'idle')
    bb.set("player.peek_expose_dir", getattr(p, '_peek_expose_dir', '') if p else '')

    # teammates
    teammates = data.get('teammate', []) or []
    tm_list = []
    for tm in teammates:
        tcx = (tm[0] + tm[2]) / 2
        tcy = (tm[1] + tm[3]) / 2
        tm_list.append({"pos": (tcx, tcy), "bbox": tm})
    bb.set("teammates", tm_list)

    # map
    walls = data.get('wall', []) or []
    bushes = getattr(p, 'last_bush_data', [])
    bb.set("map.walls", walls)
    bb.set("map.bushes", bushes)
    water_bboxes = getattr(p, '_water_bboxes', []) if p else []
    bb.set("map.water_tiles", len(water_bboxes))

    water_nearby = 0
    if water_bboxes:
        for wb in water_bboxes:
            try:
                wx = (wb[0] + wb[2]) / 2
                wy = (wb[1] + wb[3]) / 2
                if math.hypot(wx - player_pos[0], (wy - player_pos[1]) * 1.2) < 220:
                    water_nearby += 1
            except Exception:
                continue
    bb.set("map.water_nearby", water_nearby)

    # last-known enemy positions (for bush checking)
    last_known_pos = None
    if p and hasattr(p, '_get_last_known_enemy_pos'):
        last_known_pos = p._get_last_known_enemy_pos()
    bb.set("enemy.last_known_pos", last_known_pos)

    # Find dangerous bushes (near where enemies recently were)
    dangerous_bushes = []
    if p and hasattr(p, '_is_bush_dangerous'):
        for bush in bushes:
            if p._is_bush_dangerous(bush):
                bcx = (bush[0] + bush[2]) / 2
                bcy = (bush[1] + bush[3]) / 2
                dangerous_bushes.append({"pos": (bcx, bcy), "bbox": bush})
    bb.set("map.dangerous_bushes", dangerous_bushes)

    # Storm / gas
    bb.set("map.in_storm", getattr(p, '_in_storm', False))
    bb.set("map.storm_center", getattr(p, '_storm_center', (0, 0)))
    bb.set("map.storm_radius", getattr(p, '_storm_radius', 9999))
    bb.set("map.gas_active", getattr(p, '_gas_active', False))
    bb.set("map.storm_flee_delay_sec", float(getattr(p, '_storm_flee_delay_sec', 30.0) or 30.0))
    bb.set("map.spawn_side", getattr(p, '_spawn_side', None))

    # match
    bb.set("match.phase", getattr(p, '_match_phase', 'early'))
    bb.set("match.our_score", getattr(p, '_our_score', 0))
    bb.set("match.their_score", getattr(p, '_their_score', 0))
    bb.set("match.time", now)
    # match.start_time is set once per match in BTCombat.tick()
    if not bb.get("match.start_time"):
        bb.set("match.start_time", now)
    
    # Track when we last actually saw an enemy (for bush staleness)
    if enemy_list:
        bb.set("enemy.last_seen_time", now)

    # state flags
    bb.set("aggression", getattr(p, 'aggression_modifier', 1.0))

    # playstyle config
    from play import PLAYSTYLE_CONFIG
    style = PLAYSTYLE_CONFIG.get(playstyle, PLAYSTYLE_CONFIG.get("fighter", {}))
    bb.set("style", style)
    bb.set("style.hp_retreat", style.get("hp_retreat", 45))
    bb.set("style.keep_max_range", style.get("keep_max_range", False))
    bb.set("style.rush_low_enemy", style.get("rush_low_enemy", True))
    bb.set("style.approach_factor", style.get("approach_factor", 0.8))
    bb.set("style.attack_interval", style.get("attack_interval", 0.30))
    bb.set("safety.hp_critical", getattr(p, '_hp_critical_enter', 20))
    bb.set("safety.hp_conf_low", getattr(p, '_hp_conf_low_threshold', 0.45))
    bb.set("safety.hp_stale_timeout", getattr(p, '_hp_stale_timeout', 0.55))

    return bb


# --- hELPER: Smart Target Selection ---

def _pick_best_target(bb: Blackboard) -> Optional[dict]:
    """Pick the best enemy to focus on (not always the closest).
    
    OPTIMIZED Priority for Wipeout/Knockout:
    1. VERY low HP enemy (<25%) - guaranteed kill
    2. Low HP enemy (25-50%) in range - high priority finish
    3. Enemy closest to a teammate (focus fire)
    4. Closest hittable enemy
    5. Closest enemy overall
    """
    enemies = bb.get("enemies", [])
    if not enemies:
        return None
    
    attack_range = bb.get("brawler.attack_range_scaled", 300)
    ammo = bb.get("player.ammo", 0)
    attack_dmg = bb.get("brawler.attack_damage", 1200)
    teammates = bb.get("teammates", [])
    player_pos = bb.get("player.pos", (0, 0))
    
    # Pass 1: CRITICAL HP enemy (<25%) - DROP EVERYTHING and kill them
    # But only if hittable (no wall between us)
    for e in enemies:
        ehp = e.get("hp", -1)
        if 0 < ehp <= 25 and e["distance"] <= attack_range * 2.0 and e.get("hittable", True):
            return e
    
    # Pass 2: Low HP enemy (25-50%) in killable range
    for e in enemies:
        ehp = e.get("hp", -1)
        if 25 < ehp <= 50 and e["distance"] <= attack_range * 1.4 and ammo >= 1 and e.get("hittable", True):
            return e
    
    # Pass 3: Focus fire - target closest to teammates (team is already attacking)
    if teammates:
        best_focus = None
        best_focus_dist = 9999
        for e in enemies:
            if not e.get("hittable", True):
                continue
            epos = e.get("pos", (0, 0))
            # Find distance to nearest teammate
            for tm in teammates:
                tmpos = tm.get("pos", (0, 0))
                tm_dist = ((epos[0] - tmpos[0])**2 + (epos[1] - tmpos[1])**2)**0.5
                # If teammate is close to this enemy, they're probably fighting
                if tm_dist < 300 and tm_dist < best_focus_dist:
                    # Also check we can reach them
                    if e["distance"] <= attack_range * 1.5:
                        best_focus = e
                        best_focus_dist = tm_dist
        if best_focus:
            return best_focus
    
    # Pass 4: Closest hittable in range
    for e in enemies:
        if e.get("hittable", True) and e["distance"] <= attack_range:
            return e
    
    # Pass 5: Closest hittable (even out of range — reposition toward them)
    for e in enemies:
        if e.get("hittable", True):
            return e

    # Pass 6: No hittable enemy — return closest for repositioning (but bot
    # should NOT fire, the attack actions check hittable before shooting)
    return enemies[0]


def _get_optimal_range(bb: Blackboard) -> float:
    """Get the optimal fighting distance for current brawler."""
    playstyle = bb.get("brawler.playstyle", "fighter")
    attack_range = bb.get("brawler.attack_range_scaled", 300)
    safe_range = bb.get("brawler.safe_range_scaled", 200)
    
    if playstyle == "tank":
        # Tanks want to be close. Use attack_range if safe_range is 0.
        base = safe_range if safe_range > 10 else attack_range
        return base * 0.5
    elif playstyle == "assassin":
        base = safe_range if safe_range > 10 else attack_range
        return base * 0.6
    elif playstyle == "fighter":
        return attack_range * 0.65    # Fighters at medium range
    elif playstyle == "sniper":
        return attack_range * 0.85    # Snipers want max range
    elif playstyle == "thrower":
        return attack_range * 0.75    # Throwers at long-medium
    elif playstyle == "support":
        return attack_range * 0.7     # Support behind fighters
    return attack_range * 0.6


def _is_wall_blocking(player_pos, direction_keys: str, walls: list,
                      check_dist: float = 60.0) -> bool:
    """Check if a wall or water tile is blocking the intended movement direction.
    
    Projects a short ray from player_pos in the WASD direction and checks
    for intersection with wall bboxes + spatial memory grid.
    """
    if not direction_keys:
        return False
    # Convert WASD to dx, dy
    dx, dy = 0.0, 0.0
    for ch in direction_keys.upper():
        if ch == 'W': dy -= 1
        elif ch == 'S': dy += 1
        elif ch == 'A': dx -= 1
        elif ch == 'D': dx += 1
    mag = math.hypot(dx, dy)
    if mag < 0.01:
        return False
    dx, dy = dx / mag * check_dist, dy / mag * check_dist
    px, py = player_pos
    # Check endpoint against wall bboxes (expanded by margin)
    end_x, end_y = px + dx, py + dy
    margin = 30.0
    if walls:
        for w in walls:
            wx1, wy1, wx2, wy2 = w[0], w[1], w[2], w[3]
            if (wx1 - margin <= end_x <= wx2 + margin and
                    wy1 - margin <= end_y <= wy2 + margin):
                return True
    # Also check spatial memory grid for water/non-walkable cells
    global _active_spatial_memory
    sm = _active_spatial_memory
    if sm is not None:
        if not sm.is_walkable(end_x, end_y):
            return True
        # Check midpoint too
        mid_x, mid_y = px + dx * 0.5, py + dy * 0.5
        if not sm.is_walkable(mid_x, mid_y):
            return True
    return False


def _detect_concavity(player_pos, check_dist=100.0, walls=None):
    """Detect if the player is inside a U-shaped wall trap (concavity).

    Checks 4 cardinal + 4 diagonal directions for walls using BOTH
    spatial memory grid AND ONNX wall bboxes.  If 4 of 4 cardinal
    directions are blocked, returns the best escape direction.

    Probes at 50% and 100% of check_dist.  Margin is deliberately small
    to avoid false positives on scattered map walls.
    
    REDUCED check_dist from 150 to 100 to avoid false positives on maps
    with scattered walls.
    """
    if not player_pos or player_pos == (0, 0):
        return None, 0

    px, py = player_pos

    # Cardinal + diagonal probe directions  (dx, dy, key)
    DIRS = [
        ( 0, -1, 'W'),  ( 0,  1, 'S'),  (-1,  0, 'A'),  ( 1,  0, 'D'),
        (-1, -1, 'WA'), ( 1, -1, 'WD'), (-1,  1, 'SA'), ( 1,  1, 'SD'),
    ]

    # Probe at 50% and 100% - avoids triggering on distant walls
    FRACS = (0.50, 1.0)

    global _active_spatial_memory
    sm = _active_spatial_memory

    blocked = {}        # key  -> True/False
    free_dirs = []      # directions that are open

    for dx, dy, key in DIRS:
        mag = math.hypot(dx, dy)
        if mag < 0.01:
            continue
        ndx, ndy = dx / mag, dy / mag
        hit = False
        for frac in FRACS:
            dist = check_dist * frac
            ex, ey = px + ndx * dist, py + ndy * dist
            # Check ONNX wall bboxes  (tight margin to avoid false positives)
            margin = 14.0  # Reduced from 18 to be stricter
            if walls:
                for w in walls:
                    if (w[0] - margin <= ex <= w[2] + margin and
                            w[1] - margin <= ey <= w[3] + margin):
                        hit = True
                        break
            if hit:
                break
            # Check spatial memory grid (WALL / WATER / non-walkable)
            # Only trust OBSERVED cells -- unobserved ≠ blocked
            if sm is not None:
                if not sm.is_walkable(ex, ey):
                    r, c = sm.pixel_to_grid(ex, ey)
                    if 0 <= r < sm.rows and 0 <= c < sm.cols and sm.observed[r, c]:
                        hit = True
                        break
            if hit:
                break
        blocked[key] = hit
        if not hit:
            free_dirs.append(key)

    # Count cardinal blocks - REQUIRE ALL 4 to be blocked for U-trap
    cardinal_blocked = sum(1 for k in ('W', 'S', 'A', 'D') if blocked.get(k, False))

    if cardinal_blocked < 4:  # Changed from 3 to 4 - must be ALL blocked
        return None, cardinal_blocked

    # Escape: pick direction toward storm center (safe zone) if in gas
    # This prevents escaping deeper INTO the storm
    global _active_storm_center
    storm_center = getattr(_detect_concavity, '_storm_center_hint', None) or _active_storm_center
    
    # When in storm/gas, ALWAYS pick direction toward safe zone - even if walls appear blocking!
    # Escaping storm is more critical than avoiding temporary wall collision
    if storm_center:
        all_dirs = ['W', 'S', 'A', 'D', 'WA', 'WD', 'SA', 'SD']
        dirs_to_check = free_dirs if free_dirs else all_dirs  # Use all dirs if none free
        
        def _score_dir(d):
            # Movement deltas for each direction key
            deltas = {'W': (0, -1), 'S': (0, 1), 'A': (-1, 0), 'D': (1, 0),
                      'WA': (-0.7, -0.7), 'WD': (0.7, -0.7),
                      'SA': (-0.7, 0.7), 'SD': (0.7, 0.7)}
            dx, dy = deltas.get(d, (0, 0))
            # Direction toward storm center
            to_safe_x = storm_center[0] - player_pos[0]
            to_safe_y = storm_center[1] - player_pos[1]
            # Dot product: positive = moving toward safe zone
            return dx * to_safe_x + dy * to_safe_y
        
        best_dir = max(dirs_to_check, key=_score_dir)
        forced = " (FORCED)" if not free_dirs else ""
        print(f"[TRAP] Escaping toward safe zone: {best_dir}{forced} (center={int(storm_center[0])},{int(storm_center[1])})")
        return best_dir, cardinal_blocked
    
    # No storm - use free directions if available
    if free_dirs:
        # Fallback: pick the first free cardinal, else free diagonal
        for key in ('W', 'S', 'A', 'D', 'WA', 'WD', 'SA', 'SD'):
            if key in free_dirs:
                return key, cardinal_blocked

    # All 8 blocked - very tight space; try reverse of current movement
    return 'S', cardinal_blocked


def _wall_adjust_movement(player_pos, keys: str, walls: list) -> str:
    """If the intended movement hits a wall, try sliding along it.
    
    Tries the two component keys individually (e.g. 'WA' -> try 'W', try 'A'),
    then perpendicular alternatives. If all blocked, check for U-trap concavity.
    Returns original keys if nothing blocked.
    """
    if not _is_wall_blocking(player_pos, keys, walls):
        return keys  # No wall, go ahead
    
    # Try each component
    for ch in keys:
        if ch and not _is_wall_blocking(player_pos, ch, walls):
            return ch
    
    # Try perpendicular alternatives
    alts = {
        'W': ['WA', 'WD'], 'S': ['SA', 'SD'],
        'A': ['WA', 'SA'], 'D': ['WD', 'SD'],
        'WA': ['W', 'A'], 'WD': ['W', 'D'],
        'SA': ['S', 'A'], 'SD': ['S', 'D'],
    }
    for alt in alts.get(keys, []):
        if not _is_wall_blocking(player_pos, alt, walls):
            return alt
    
    # All standard directions blocked - check if we're in a U-trap
    escape, blocked_n = _detect_concavity(player_pos, walls=walls)
    if escape and blocked_n >= 4:
        # Throttle print: only log every 5 seconds
        global _last_trap_print_time
        if time.time() - _last_trap_print_time > 5.0:
            print(f"[TRAP] U-shape detected ({blocked_n}/4 blocked) -> escape {escape}")
            _last_trap_print_time = time.time()
        return escape
    
    return keys  # Give up, keep original


def _compute_movement_toward(player_pos, target_pos, strafe_amount=0.0,
                              strafe_dir=1, walls=None, pathfinder=None, spatial_memory=None):
    """Compute WASD movement toward a target with optional strafe component.
    
    strafe_amount: 0.0 = straight at target, 1.0 = full perpendicular strafe
    strafe_dir: 1 or -1 for left/right strafe
    walls: optional list of [x1,y1,x2,y2] wall bboxes for collision avoidance
    pathfinder: optional PathPlanner (falls back to module-level _active_pathfinder)
    spatial_memory: optional SpatialMemory (falls back to module-level _active_spatial_memory)
    """
    global _active_pathfinder, _active_spatial_memory
    if pathfinder is None:
        pathfinder = _active_pathfinder
    if spatial_memory is None:
        spatial_memory = _active_spatial_memory
    
    dx = target_pos[0] - player_pos[0]
    dy = target_pos[1] - player_pos[1]
    mag = math.hypot(dx, dy)
    if mag < 1:
        return ""  # Already at target
    
    # Normalize
    ndx, ndy = dx / mag, dy / mag
    
    # Add perpendicular strafe
    if strafe_amount > 0:
        perp_x = -ndy * strafe_dir * strafe_amount
        perp_y = ndx * strafe_dir * strafe_amount
        ndx = ndx * (1 - strafe_amount) + perp_x
        ndy = ndy * (1 - strafe_amount) + perp_y
    
    # Convert to WASD
    h = 'D' if ndx > 0.2 else ('A' if ndx < -0.2 else '')
    v = 'S' if ndy > 0.2 else ('W' if ndy < -0.2 else '')
    keys = (v + h).upper() or "W"
    
    # PRIMARY: Use A* pathfinder when available (avoids walls proactively)
    if pathfinder is not None and spatial_memory is not None:
        try:
            pf_keys = pathfinder.get_movement_toward(
                player_pos, target_pos, spatial_memory
            )
            if pf_keys:
                return pf_keys
        except Exception:
            pass
    
    # Check for U-trap concavity BEFORE simple wall adjustment
    escape, blocked_n = _detect_concavity(player_pos, walls=walls)
    if escape and blocked_n >= 4:
        # Throttle print: only log every 5 seconds
        global _last_concavity_print_time
        if time.time() - _last_concavity_print_time > 5.0:
            print(f"[TRAP] Concavity ({blocked_n}/4 blocked) - escaping {escape} instead of {keys}")
            _last_concavity_print_time = time.time()
        return escape
    
    # FALLBACK: Wall-adjusted direct movement
    if walls:
        adjusted = _wall_adjust_movement(player_pos, keys, walls)
        return adjusted
    return keys


def _compute_movement_away(player_pos, threat_pos, strafe_amount=0.3,
                            strafe_dir=1, walls=None, pathfinder=None, spatial_memory=None):
    """Compute movement AWAY from a threat with strafe dodge."""
    global _active_pathfinder, _active_spatial_memory
    if pathfinder is None:
        pathfinder = _active_pathfinder
    if spatial_memory is None:
        spatial_memory = _active_spatial_memory
    
    dx = player_pos[0] - threat_pos[0]  # inverted = away
    dy = player_pos[1] - threat_pos[1]
    mag = math.hypot(dx, dy)
    if mag < 1:
        dx, dy = 0, -1  # Default: flee up
        mag = 1
    
    ndx, ndy = dx / mag, dy / mag
    
    # Add strafe dodge
    if strafe_amount > 0:
        perp_x = -ndy * strafe_dir * strafe_amount
        perp_y = ndx * strafe_dir * strafe_amount
        ndx += perp_x
        ndy += perp_y
    
    h = 'D' if ndx > 0.2 else ('A' if ndx < -0.2 else '')
    v = 'S' if ndy > 0.2 else ('W' if ndy < -0.2 else '')
    keys = (v + h).upper() or "W"
    
    # PRIMARY: Use A* pathfinder for flee when available
    if pathfinder is not None and spatial_memory is not None:
        try:
            pf_keys = pathfinder.get_movement_away(
                player_pos, threat_pos, spatial_memory
            )
            if pf_keys:
                return pf_keys
        except Exception:
            pass
    
    # FALLBACK: Wall-adjusted direct movement
    if walls:
        adjusted = _wall_adjust_movement(player_pos, keys, walls)
        return adjusted
    return keys


# --- cONDITION NODES ---

def _storm_threat_active(bb: Blackboard) -> bool:
    """Effective storm/gas threat with optional match-time delay gating."""
    active = bool(bb.get("map.in_storm", False)) or bool(bb.get("map.gas_active", False))
    if not active:
        return False

    start_time = bb.get("match.start_time", None)
    now = bb.get("match.time", None)
    if start_time is None or now is None:
        return active

    try:
        elapsed = float(now) - float(start_time)
    except Exception:
        return active

    delay_sec = float(bb.get("map.storm_flee_delay_sec", 30.0) or 30.0)
    return elapsed >= max(0.0, delay_sec)

def cond_in_storm(bb: Blackboard) -> bool:
    return _storm_threat_active(bb)

def cond_no_enemies(bb: Blackboard) -> bool:
    return bb.get("enemies_count", 0) == 0

def cond_is_dead(bb: Blackboard) -> bool:
    return bb.get("player.is_dead", False)

def cond_low_hp(bb: Blackboard) -> bool:
    hp = bb.get("player.hp", 100)
    # style.hp_retreat is ALREADY adjusted for aggression in BTCombat.tick()
    # - no further division by aggression here (was causing double-penalty)
    threshold = bb.get("style.hp_retreat", 45)
    result = hp < threshold and hp > 0
    # Diagnostic: log when retreat triggers so we can verify thresholds
    if result:
        import time as _t
        if not hasattr(cond_low_hp, '_last_log') or _t.time() - cond_low_hp._last_log > 2.0:
            print(f"[BT-RETREAT] hp={hp} < threshold={threshold} -> retreat")
            cond_low_hp._last_log = _t.time()
    return result

def cond_enemy_killable(bb: Blackboard) -> bool:
    """Enemy HP is low enough to finish with remaining ammo.
    
    OPTIMIZED: More aggressive kill chasing - finish enemies up to 65% HP
    and chase them further for kill secure.
    """
    target = _pick_best_target(bb)
    if not target:
        return False
    ehp = target.get("hp", 100)
    if ehp <= 0 or ehp > 65:  # Raised from 55→65: finish enemies MUCH more aggressively
        return False
    ammo = bb.get("player.ammo", 0)
    attack_dmg = bb.get("brawler.attack_damage", 1200)
    # Use actual brawler health for more accurate estimation
    enemy_base_hp = bb.get("brawler.health", 3200)  # fallback if unknown
    enemy_abs_hp = enemy_base_hp * (ehp / 100.0)
    shots_needed = max(1, math.ceil(enemy_abs_hp / max(1, attack_dmg)))
    
    # Chase threshold: Chase low HP enemies further than normal attack range
    distance = target.get("distance", 999)
    attack_range = bb.get("brawler.attack_range_scaled", 300)
    # If VERY low HP (<35%), chase even further (1.8x range)
    # If moderately low (35-65%), chase at 1.4x range  
    chase_mult = 1.8 if ehp < 35 else 1.4
    max_chase_dist = attack_range * chase_mult
    
    if distance > max_chase_dist:
        return False  # Too far to chase
    
    # Accept if we can kill OR if enemy is low enough that chasing is worth it
    if ammo >= shots_needed or ehp < 30:
        bb.set("_target_enemy", target)  # Store chosen target
        return True
    return False

def cond_enemy_in_bush(bb: Blackboard) -> bool:
    """True if no enemies visible but we recently saw one near a bush.
    
    This means the enemy likely ran into a bush and is hiding.
    We should shoot at the bush to check / flush them out.
    Strict conditions to avoid wasting ammo at round start.
    """
    if bb.get("enemies_count", 0) > 0:
        return False  # We can still see enemies, no need to bush-check
    
    # Grace period: don't bush-check in first 5s of match (no info yet)
    match_time = bb.get("match.time", 0)
    match_start = bb.get("match.start_time", 0)
    if match_time - match_start < 5.0:
        return False
    
    # Need FULL ammo to shoot bush (don't waste ammo needed for combat)
    if bb.get("player.ammo", 0) < 3:
        return False  # Keep ammo for actual fights
    
    # Need a last-known position
    last_known = bb.get("enemy.last_known_pos")
    if not last_known:
        return False
    
    # Last-known position must be recent (< 3s old, not stale from previous round)
    last_seen_time = bb.get("enemy.last_seen_time", 0)
    if match_time - last_seen_time > 3.0:
        return False  # Too old, enemy could be anywhere
    
    # Must be close enough to actually hit the bush
    player_pos = bb.get("player.pos", (0, 0))
    dist = math.hypot(last_known[0] - player_pos[0],
                       (last_known[1] - player_pos[1]) * 1.25)
    atk_range = bb.get("brawler.attack_range_scaled", 300)
    if dist > atk_range * 1.3:
        return False  # Too far to hit anyway
    
    # Check if there's a dangerous bush (enemy was seen near it)
    dangerous = bb.get("map.dangerous_bushes", [])
    if dangerous:
        return True
    
    # Without detected bushes, only shoot if enemy JUST vanished (<1.5s)
    if match_time - last_seen_time < 1.5:
        return True
    
    return False


def cond_enemy_in_range(bb: Blackboard) -> bool:
    closest = bb.get("enemies_closest")
    if not closest:
        return False
    return closest["distance"] <= bb.get("brawler.attack_range_scaled", 300)

def cond_enemy_hittable(bb: Blackboard) -> bool:
    closest = bb.get("enemies_closest")
    if not closest:
        return False
    return closest.get("hittable", False)


def _should_block_attack_for_hp_safety(bb: Blackboard) -> Tuple[bool, str]:
    """Global safety gate for firing decisions under uncertain HP tracking.

    Policy:
    - If HP is unknown/stale and confidence is low -> block attack
    - If HP is critical and confidence is low -> block attack
    """
    hp = int(bb.get("player.hp", 100) or 100)
    hp_conf = float(bb.get("player.hp_confidence", 1.0) or 0.0)
    hp_state = str(bb.get("player.hp_state", "healthy") or "healthy").lower()
    hp_age = float(bb.get("player.hp_age", 0.0) or 0.0)
    critical_hp = int(bb.get("safety.hp_critical", 20) or 20)
    conf_low = float(bb.get("safety.hp_conf_low", 0.45) or 0.45)
    stale_timeout = float(bb.get("safety.hp_stale_timeout", 0.55) or 0.55)

    closest = bb.get("enemies_closest")
    attack_range = float(bb.get("brawler.attack_range_scaled", 300) or 300)
    has_ammo = int(bb.get("player.ammo", 0) or 0) >= 1
    close_hittable_enemy = False
    if closest:
        close_hittable_enemy = bool(closest.get("hittable", False)) and float(closest.get("distance", 9999) or 9999) <= attack_range * 1.05

    # In close, hittable duels with ammo, don't over-block on uncertain HP state
    # unless we're truly critical.
    if close_hittable_enemy and has_ammo and hp > max(12, critical_hp - 6):
        return False, ""

    if hp_state == "unknown" and (hp_conf < conf_low and hp_age > stale_timeout):
        return True, "HP unknown+stale"
    if hp <= critical_hp and hp_conf < conf_low:
        return True, "HP critical+uncertain"
    return False, ""

def cond_has_ammo(bb: Blackboard) -> bool:
    return bb.get("player.ammo", 0) >= 1

def cond_super_ready(bb: Blackboard) -> bool:
    return bb.get("player.super_ready", False)

def cond_should_use_super(bb: Blackboard) -> bool:
    """Smart super usage: use aggressively whenever enemies are in range.
    
    Super is valuable but should be used regularly, not hoarded.
    Use it as soon as we have a reasonable target.
    """
    if not bb.get("player.super_ready", False):
        return False
    # BT-level cooldown: don't spam super
    now = bb.get("match.time", 0)
    last_super = bb.get("_last_super_time", 0)
    if now - last_super < 1.5:  # 1.5s cooldown between super attempts
        return False
    closest = bb.get("enemies_closest")
    if not closest:
        return False
    
    distance = closest.get("distance", 999)
    super_range = bb.get("brawler.super_range_scaled", 300)
    enemies_count = bb.get("enemies_count", 0)
    playstyle = bb.get("brawler.playstyle", "fighter")
    hp = bb.get("player.hp", 100)
    ammo = bb.get("player.ammo", 0)
    
    # Don't super if no enemy in super range (generous margin)
    if distance > super_range * 1.3:
        return False
    
    # === Tanks / Fighters / Assassins: use aggressively in close combat ===
    if playstyle in ("fighter", "tank", "assassin"):
        ehp = closest.get("hp", 100)
        # Low HP enemy = finish them
        if ehp <= 60 and distance < super_range * 1.0:
            return True
        # Multiple enemies close = area damage
        if enemies_count >= 2 and distance < super_range * 0.9:
            return True
        # In combat range with ammo = combo opportunity
        if distance < super_range * 0.7 and ammo >= 1:
            return True
        # Defensive: low HP and enemy close = try to burst them
        if hp < 60 and distance < super_range * 0.8:
            return True
    
    # === Ranged: snipers / throwers ===
    elif playstyle in ("sniper", "thrower"):
        if enemies_count >= 2 and distance < super_range * 1.0:
            return True
        if closest.get("hp", 100) <= 60:
            return True
        # Use when in good range and have ammo for follow-up
        if distance < super_range * 0.9 and ammo >= 1:
            return True
    
    # === Support: self-heal or area control ===
    elif playstyle == "support":
        if hp < 60:  # Self-heal threshold
            return True
        if enemies_count >= 2 and distance < super_range * 0.9:
            return True
        # Use in combat
        if distance < super_range * 0.8 and ammo >= 1:
            return True
    
    # === Any playstyle: enemy in super range -> just use it ===
    if distance < super_range * 1.0:
        return True
    
    # === Fallback: haven't used in 5s and enemy is anywhere near ===
    if now - last_super > 5.0 and distance < super_range * 1.2:
        return True
    
    return False

def cond_should_use_gadget(bb: Blackboard) -> bool:
    """Smart gadget usage: use aggressively in any combat situation.
    
    Gadgets recharge over time, so use them whenever they're available
    and any enemy is visible. Don't hoard them.
    """
    if not bb.get("player.gadget_ready", False):
        return False
    closest = bb.get("enemies_closest")
    if not closest:
        return False
    
    # BT-level cooldown: prevent spamming (gadgets have ~8s game cooldown anyway)
    now = bb.get("match.time", 0)
    last_gadget = bb.get("_last_gadget_time", 0)
    if now - last_gadget < 2.0:  # 2s minimum between attempts
        return False
    
    hp = bb.get("player.hp", 100)
    distance = closest.get("distance", 999)
    attack_range = bb.get("brawler.attack_range_scaled", 300)
    enemies_count = bb.get("enemies_count", 0)
    
    # Defensive: HP not full and enemy is nearby
    if hp < 70 and distance < attack_range * 1.5:
        return True
    # Enemy in attack range = use it
    if distance < attack_range * 1.0 and enemies_count >= 1:
        return True
    # Multiple enemies visible
    if enemies_count >= 2:
        return True
    # Haven't used gadget in a while and enemy is anywhere near
    if now - last_gadget > 5.0 and distance < attack_range * 1.2:
        return True
    return False

def cond_enemy_far_and_aggressive(bb: Blackboard) -> bool:
    """True if enemy is out of range AND we should close distance.
    
    OPTIMIZED: Don't approach with 0 ammo unless enemy is very low HP.
    """
    closest = bb.get("enemies_closest")
    if not closest:
        return False
    distance = closest.get("distance", 0)
    attack_range = bb.get("brawler.attack_range_scaled", 300)
    ammo = bb.get("player.ammo", 0)
    
    if distance < attack_range * 1.1:
        return False
    
    # Don't approach with 0 ammo - wait for at least 1 shot
    # Exception: If enemy is VERY low HP, chase anyway
    enemy_hp = closest.get("hp", 100)
    if ammo == 0 and enemy_hp > 30:
        return False  # Wait for reload
    
    # ALL playstyles should approach when ready
    return True

def cond_enemy_out_of_range(bb: Blackboard) -> bool:
    closest = bb.get("enemies_closest")
    if not closest:
        return False
    distance = closest.get("distance", 0)
    attack_range = bb.get("brawler.attack_range_scaled", 300)
    return distance > attack_range * 1.1

def cond_respawn_shield(bb: Blackboard) -> bool:
    return bb.get("player.respawn_shield", False)

def cond_keep_max_range(bb: Blackboard) -> bool:
    return bb.get("style.keep_max_range", False)

def cond_too_close(bb: Blackboard) -> bool:
    """True if enemy is dangerously close for a ranged brawler.
    Throwers excluded - they can fight close with splash damage."""
    closest = bb.get("enemies_closest")
    if not closest:
        return False
    playstyle = bb.get("brawler.playstyle", "fighter")
    if playstyle not in ("sniper", "support"):
        return False  # Throwers removed - they should fight, not kite!
    safe_range = bb.get("brawler.safe_range_scaled", 200)
    return closest["distance"] < safe_range * 0.6


def cond_should_disengage_heal(bb: Blackboard) -> bool:
    """True if we should disengage to regenerate HP.
    
    Triggers when:
    - HP is between 20-55% (hurt enough to want regen)
    - Enemy is far enough (> safe_range * 0.4) to safely retreat
    - Not in late match phase (can't afford to wait)
    - Not already in disengage cooldown
    """
    hp = bb.get("player.hp", 100)
    if hp <= 15 or hp >= 50:
        return False  # Too low (use retreat) or healthy enough to fight
    
    phase = bb.get("match.phase", "early")
    if phase == "late":
        return False  # Can't afford to disengage late game
    
    closest = bb.get("enemies_closest")
    if not closest:
        return False  # No enemy to disengage from

    attack_range = float(bb.get("brawler.attack_range_scaled", 300) or 300)
    in_attack_window = float(closest.get("distance", 0) or 0) <= attack_range * 1.1
    if in_attack_window and bool(closest.get("hittable", False)) and int(bb.get("player.ammo", 0) or 0) >= 1:
        return False  # Keep fighting when enemy is clearly punishable
    
    safe_range = bb.get("brawler.safe_range_scaled", 200)
    if closest["distance"] < safe_range * 0.5:
        return False  # Too close, retreat instead (handled by combat tree)
    
    # Cooldown: don't disengage again within 3s
    last_disengage = bb.get("_last_disengage_end", 0)
    now = bb.get("match.time", 0)
    if now - last_disengage < 3.0:
        return False
    
    return True


def cond_taking_damage(bb: Blackboard) -> bool:
    """True if player just took significant damage this frame.
    
    Used to trigger reactive dodge: briefly move perpendicular to attacker.
    OPTIMIZED: Lower threshold and faster cooldown for more responsive dodging.
    """
    damage = bb.get("player.damage_taken", 0)
    if damage < 300:  # Lowered from 500→300: dodge even on lighter hits (~10% HP)
        return False
    # Grace period: no dodge in first 2s of match (was 3s)
    now = bb.get("match.time", 0)
    match_start = bb.get("match.start_time", 0)
    if match_start and now - match_start < 2.0:
        return False
    # Don't re-trigger if already dodging
    dodge_until = bb.get("_reactive_dodge_until", 0)
    if now <= dodge_until:
        return False
    # Extra safety: at least 0.5s between dodge triggers (was 1.0s)
    last_dodge = bb.get("_last_dodge_trigger", 0)
    if now - last_dodge < 0.5:
        return False
    bb.set("_last_dodge_trigger", now)
    return True

def cond_enemy_in_optimal_range(bb: Blackboard) -> bool:
    """True if closest enemy is near our optimal fighting range."""
    closest = bb.get("enemies_closest")
    if not closest:
        return False
    optimal = _get_optimal_range(bb)
    distance = closest["distance"]
    tolerance = optimal * 0.25
    return abs(distance - optimal) <= tolerance


# --- aCTION NODES ---

def _update_strafe(bb: Blackboard):
    """Manage committed strafe direction with distance-aware combat patterns."""
    now = float(bb.get("match.time", 0.0) or 0.0)
    if now <= 0:
        now = time.time()

    closest = bb.get("enemies_closest") or {}
    distance = float(closest.get("distance", 999.0) or 999.0)
    attack_range = float(bb.get("brawler.attack_range_scaled", 300.0) or 300.0)

    # Keep one strafe side for longer windows to avoid left-right jitter.
    # Closer enemies => shorter commits, medium range => longer duel commits.
    if distance < attack_range * 0.75:
        commit_dur = random.uniform(0.35, 0.60)
    elif distance < attack_range * 1.8:
        commit_dur = random.uniform(0.55, 0.95)
    else:
        commit_dur = random.uniform(0.45, 0.80)

    strafe_until = float(bb.get("_strafe_until", 0.0) or 0.0)
    strafe_dir = int(bb.get("_strafe_dir", 1) or 1)
    last_switch = float(bb.get("_strafe_last_switch", 0.0) or 0.0)
    switch_cooldown = 0.22
    can_switch = (now - last_switch) >= switch_cooldown
    if now >= strafe_until:
        # Bias to side switch when enemy pressure is high to break aim locks.
        pattern_pressure = float(bb.get("enemy.pattern_pressure", 0.0) or 0.0)
        attack_soon = float(bb.get("enemy.predicted_attack_soon", 0.0) or 0.0)
        if can_switch and attack_soon > 0.5 and random.random() < 0.70:
            strafe_dir = -strafe_dir
        elif can_switch and pattern_pressure > 0.65 and random.random() < 0.55:
            strafe_dir = -strafe_dir
        elif can_switch and random.random() < 0.25:
            strafe_dir = -strafe_dir
        bb.set("_strafe_dir", 1 if strafe_dir >= 0 else -1)
        bb.set("_strafe_until", now + commit_dur)
        bb.set("_strafe_last_switch", now)
    return int(bb.get("_strafe_dir", 1) or 1)


def _update_flank_dir(bb: Blackboard) -> int:
    """Committed flank direction for multi-step approach routes."""
    now = float(bb.get("match.time", 0.0) or 0.0)
    if now <= 0:
        now = time.time()
    flank_until = float(bb.get("_flank_until", 0.0) or 0.0)
    flank_dir = int(bb.get("_flank_dir", 1) or 1)
    if now >= flank_until:
        if random.random() < 0.50:
            flank_dir = -flank_dir
        bb.set("_flank_dir", 1 if flank_dir >= 0 else -1)
        bb.set("_flank_until", now + random.uniform(1.0, 1.9))
    return int(bb.get("_flank_dir", 1) or 1)


def _compute_flank_target(player_pos, enemy_pos, attack_range, flank_dir=1):
    """Compute a lateral flank waypoint near enemy for less linear engagements."""
    dx = enemy_pos[0] - player_pos[0]
    dy = enemy_pos[1] - player_pos[1]
    mag = math.hypot(dx, dy)
    if mag < 1:
        return enemy_pos
    ndx, ndy = dx / mag, dy / mag
    # Perpendicular offset gives orbit/flank path; slight backward bias avoids overcommitting.
    perp_x = -ndy * flank_dir
    perp_y = ndx * flank_dir
    side_offset = max(120.0, min(260.0, attack_range * 0.75))
    back_bias = min(120.0, attack_range * 0.25)
    return (
        enemy_pos[0] + perp_x * side_offset - ndx * back_bias,
        enemy_pos[1] + perp_y * side_offset - ndy * back_bias,
    )


def act_flee_storm(bb: Blackboard) -> Status:
    """Move toward storm center (safe zone)."""
    player_pos = bb.get("player.pos", (0, 0))
    center = bb.get("map.storm_center", (0, 0))
    walls = bb.get("map.walls", [])
    movement = _compute_movement_toward(player_pos, center, walls=walls)
    bb.set("decision.movement", movement)
    bb.set("decision.reason", "BT: FLEE STORM")
    return Status.SUCCESS

def act_wait_dead(bb: Blackboard) -> Status:
    bb.set("decision.movement", "")
    bb.set("decision.reason", "BT: DEAD")
    return Status.SUCCESS

def act_patrol(bb: Blackboard) -> Status:
    """No enemies - hunt them down actively.
    
    Priority:
    1. Chase last-known enemy position (if recent)
    2. Move toward teammates if far apart
    3. Push toward map center / enemy side
    """
    playstyle = bb.get("brawler.playstyle", "fighter")
    player_pos = bb.get("player.pos", (0, 0))
    teammates = bb.get("teammates", [])
    walls = bb.get("map.walls", [])
    
    # --- PRIORITY 1: Chase last-known enemy position ---
    # Only chase if we saw enemies recently (within 5s). Stale positions mean
    # the enemy is long gone and we'd just get stuck heading toward a wall.
    last_known = bb.get("enemy.last_known_pos")
    if last_known:
        now = bb.get("match.time", 0)
        last_seen = bb.get("enemy.last_seen_time", 0)
        time_since_enemy = now - last_seen if last_seen else 999
        dist_to_last = math.hypot(last_known[0] - player_pos[0],
                                   last_known[1] - player_pos[1])
        if dist_to_last > 60 and time_since_enemy < 5.0:  # Give up after 5s
            movement = _compute_movement_toward(player_pos, last_known,
                                                strafe_amount=0.1, walls=walls)
            bb.set("decision.movement", movement)
            bb.set("decision.reason", "BT: HUNT LAST POS")
            return Status.SUCCESS
        elif time_since_enemy >= 5.0:
            # Stale position - clear it so we stop hunting
            bb.set("enemy.last_known_pos", None)
    
    # --- PRIORITY 2: Stay near teammates (but not too cautious) ---
    if teammates and playstyle not in ("assassin",):
        tm_x = sum(t["pos"][0] for t in teammates) / len(teammates)
        tm_y = sum(t["pos"][1] for t in teammates) / len(teammates)
        dist_to_team = math.hypot(tm_x - player_pos[0], tm_y - player_pos[1])
        if dist_to_team > 300:  # Only regroup if VERY far
            movement = _compute_movement_toward(player_pos, (tm_x, tm_y),
                                                strafe_amount=0.15, walls=walls)
            bb.set("decision.movement", movement)
            bb.set("decision.reason", "BT: REGROUP")
            return Status.SUCCESS
    
    # --- BT PATROL ANTI-STUCK: reset patrol if position hasn't changed ---
    _prev_patrol_pos = bb.get("_patrol_prev_pos")
    _patrol_stuck_count = int(bb.get("_patrol_stuck_count", 0) or 0)
    if _prev_patrol_pos:
        moved = math.hypot(player_pos[0] - _prev_patrol_pos[0],
                           player_pos[1] - _prev_patrol_pos[1])
        if moved < 15:
            _patrol_stuck_count += 1
        else:
            _patrol_stuck_count = 0
    bb.set("_patrol_prev_pos", player_pos)
    bb.set("_patrol_stuck_count", _patrol_stuck_count)

    # If stuck for 8+ ticks (~0.5-1s), force-pick a new patrol target
    if _patrol_stuck_count >= 8:
        bb.set("_patrol_tick", 0)  # force target re-pick
        bb.set("_patrol_stuck_count", 0)

    # --- PRIORITY 3: Spawn-aware full-map search (forward-biased) ---
    tick = int(bb.get("_patrol_tick", 0) or 0)
    patrol_dir = bb.get("_patrol_dir", "W")
    spawn_side = bb.get("map.spawn_side", None)

    if spawn_side == "left":
        fx, fy = 1.0, 0.0
    elif spawn_side == "right":
        fx, fy = -1.0, 0.0
    elif spawn_side == "bottom":
        fx, fy = 0.0, -1.0
    elif spawn_side == "top":
        fx, fy = 0.0, 1.0
    else:
        fx, fy = 0.0, -1.0
    lx, ly = -fy, fx

    match_time = bb.get("match.time", 0)
    match_start = bb.get("match.start_time", 0)
    elapsed = match_time - match_start

    if elapsed < 10.0:
        # Push forward aggressively for the first 10 seconds of a round.
        # This ensures the bot leaves spawn and reaches the mid-map area
        # where enemies are likely to be.
        push_dist = 320 if elapsed < 3.0 else 200
        fwd_target = (
            max(80, min(1840, player_pos[0] + fx * push_dist)),
            max(80, min(990, player_pos[1] + fy * push_dist)),
        )
        movement = _compute_movement_toward(player_pos, fwd_target,
                                             strafe_amount=0.0, walls=walls)
        bb.set("decision.movement", movement)
        bb.set("decision.reason", "BT: PUSH FORWARD (start)")
        return Status.SUCCESS

    if tick <= 0:
        offsets_all = [
            (260, 0), (230, 140), (230, -140),
            (330, 40), (330, -40),
            (180, 180), (180, -180),
            (120, 0),
        ]
        random.shuffle(offsets_all)
        patrol_target = None
        sm = _active_spatial_memory
        for fwd_off, lat_off in offsets_all:
            tx = max(80, min(1840, player_pos[0] + fx * fwd_off + lx * lat_off))
            ty = max(80, min(990, player_pos[1] + fy * fwd_off + ly * lat_off))
            blocked = False
            if sm is not None and not sm.is_walkable(tx, ty):
                blocked = True
            # NOTE: midpoint walkability check removed — it was too strict
            # and caused all patrol targets to be rejected when ghost-walls
            # from colour detection existed between player and target.
            if not blocked and walls:
                margin = 15.0
                for w in walls:
                    if (w[0] - margin <= tx <= w[2] + margin and
                            w[1] - margin <= ty <= w[3] + margin):
                        blocked = True
                        break
            if not blocked:
                patrol_target = (tx, ty)
                break

        if patrol_target is None:
            # Fallback: push forward instead of going to static (960,540)
            patrol_target = (
                max(80, min(1840, player_pos[0] + fx * 200)),
                max(80, min(990, player_pos[1] + fy * 200)),
            )

        patrol_dir = _compute_movement_toward(player_pos, patrol_target,
                                               strafe_amount=0.05, walls=walls)
        bb.set("_patrol_dir", patrol_dir)
        bb.set("_patrol_target", patrol_target)
        bb.set("_patrol_tick", random.randint(18, 42))
    else:
        bb.set("_patrol_tick", tick - 1)
        stored_target = bb.get("_patrol_target")
        if stored_target:
            if math.hypot(stored_target[0] - player_pos[0], stored_target[1] - player_pos[1]) < 90:
                bb.set("_patrol_tick", 0)
            patrol_dir = _compute_movement_toward(player_pos, stored_target,
                                                   strafe_amount=0.05, walls=walls)
        else:
            patrol_dir = bb.get("_patrol_dir", "W")

    bb.set("decision.movement", patrol_dir)
    bb.set("decision.reason", "BT: SEARCH MAP")
    return Status.SUCCESS

def act_retreat(bb: Blackboard) -> Status:
    """Low HP - smart retreat with dodge pattern and counter-fire.
    
    OPTIMIZED: Retreat toward teammates for protection, higher strafe for survival.
    """
    player_pos = bb.get("player.pos", (0, 0))
    closest = bb.get("enemies_closest")
    strafe_dir = _update_strafe(bb)
    teammates = bb.get("teammates", [])
    
    walls = bb.get("map.walls", [])
    hp = bb.get("player.hp", 100)
    
    # Find nearest teammate to retreat toward (safety in numbers)
    retreat_target = None
    if teammates:
        nearest_tm = None
        nearest_dist = 9999
        for tm in teammates:
            tmpos = tm.get("pos", (0, 0))
            d = ((tmpos[0] - player_pos[0])**2 + (tmpos[1] - player_pos[1])**2)**0.5
            if d < nearest_dist and d > 100:  # Don't stack on top of teammate
                nearest_dist = d
                nearest_tm = tmpos
        if nearest_tm and nearest_dist < 600:
            retreat_target = nearest_tm
    
    if closest:
        enemy_pos = closest["pos"]
        distance = closest.get("distance", 999)
        
        # Higher strafe when very low HP (50% strafe for survival)
        strafe_amt = 0.50 if hp < 30 else 0.40
        
        if retreat_target:
            # Retreat TOWARD teammate (not just away from enemy)
            # Blend: 70% toward teammate, 30% away from enemy
            dx_tm = retreat_target[0] - player_pos[0]
            dy_tm = retreat_target[1] - player_pos[1]
            dx_away = player_pos[0] - enemy_pos[0]
            dy_away = player_pos[1] - enemy_pos[1]
            
            # Normalize and blend
            mag_tm = max(1, (dx_tm**2 + dy_tm**2)**0.5)
            mag_away = max(1, (dx_away**2 + dy_away**2)**0.5)
            
            blend_x = 0.7 * (dx_tm / mag_tm) + 0.3 * (dx_away / mag_away)
            blend_y = 0.7 * (dy_tm / mag_tm) + 0.3 * (dy_away / mag_away)
            
            # Convert to movement keys
            h = 'D' if blend_x > 0.2 else ('A' if blend_x < -0.2 else '')
            v = 'S' if blend_y > 0.2 else ('W' if blend_y < -0.2 else '')
            movement = (v + h).upper() or "W"
        else:
            # No teammate nearby - standard retreat away from enemy
            movement = _compute_movement_away(player_pos, enemy_pos, 
                                               strafe_amount=strafe_amt, strafe_dir=strafe_dir,
                                               walls=walls)
        
        # Counter-fire while retreating if we have ammo and in range
        attack_range = bb.get("brawler.attack_range_scaled", 400)
        has_ammo = bb.get("player.ammo", 0) >= 1
        
        # Fire while retreating - every hit counts even when running
        can_fire = has_ammo and distance <= attack_range and closest.get("hittable", True)
        bb.set("decision.should_attack", can_fire)
        attack_str = "+FIRE" if can_fire else ""
        tm_str = "->TM" if retreat_target else ""
        bb.set("decision.reason", f"BT: RETREAT{tm_str}{attack_str} (HP {hp}%)")
    else:
        movement = "W"  # Flee up by default
        bb.set("decision.should_attack", False)
        bb.set("decision.reason", f"BT: RETREAT (HP {hp}%)")
    
    bb.set("decision.movement", movement)
    return Status.SUCCESS

def act_finish_kill(bb: Blackboard) -> Status:
    """Rush and kill a low-HP enemy - all-in aggression.
    
    OPTIMIZED: More aggressive pursuit with predictive lead compensation.
    """
    player_pos = bb.get("player.pos", (0, 0))
    target = bb.get("_target_enemy") or bb.get("enemies_closest")
    if not target:
        return Status.FAILURE
    
    distance = target["distance"]
    strafe_dir = _update_strafe(bb)
    
    walls = bb.get("map.walls", [])
    enemy_pos = target["pos"]
    
    # Predict enemy movement and intercept
    enemy_vel = bb.get("enemy.velocity", (0, 0))
    if enemy_vel and (enemy_vel[0] != 0 or enemy_vel[1] != 0):
        # Lead target: predict where enemy will be in ~0.3s
        lead_time = min(0.4, distance / 600)  # Scale lead with distance
        predicted_pos = (
            enemy_pos[0] + enemy_vel[0] * lead_time,
            enemy_pos[1] + enemy_vel[1] * lead_time
        )
        enemy_pos = predicted_pos
    
    # Rush directly at them with aggressive strafe (higher when close)
    strafe_amt = 0.25 if distance > 200 else 0.1  # Less strafe when close = more direct
    movement = _compute_movement_toward(player_pos, enemy_pos, 
                                         strafe_amount=strafe_amt, strafe_dir=strafe_dir, walls=walls)
    
    hittable = target.get("hittable", True)
    has_ammo = bb.get("player.ammo", 0) >= 1
    attack_range = bb.get("brawler.attack_range_scaled", 300)
    
    # Fire when in range (with margin for movement)
    in_attack_range = distance <= attack_range * 1.15
    
    bb.set("decision.movement", movement)
    bb.set("decision.should_attack", has_ammo and hittable and in_attack_range)
    bb.set("decision.attack_type", "auto")
    hit_str = "+FIRE" if (has_ammo and hittable and in_attack_range) else "+CHASE"
    bb.set("decision.reason", f"BT: FINISH KILL{hit_str} ({target.get('hp', '?')}% @ {int(distance)}px)")
    return Status.SUCCESS

def act_use_super(bb: Blackboard) -> Status:
    """Use super ability."""
    bb.set("decision.use_super", True)
    bb.set("_last_super_time", bb.get("match.time", 0))  # Record for cooldown
    bb.set("decision.reason", "BT: USE SUPER")
    return Status.SUCCESS

def act_use_gadget(bb: Blackboard) -> Status:
    """Use gadget ability."""
    bb.set("decision.use_gadget", True)
    bb.set("_last_gadget_time", bb.get("match.time", 0))  # Record for cooldown
    bb.set("decision.reason", "BT: USE GADGET")
    return Status.SUCCESS

def act_kite_back(bb: Blackboard) -> Status:
    """Enemy too close for ranged brawler - retreat while firing."""
    player_pos = bb.get("player.pos", (0, 0))
    closest = bb.get("enemies_closest")
    if not closest:
        return Status.FAILURE
    
    walls = bb.get("map.walls", [])
    strafe_dir = _update_strafe(bb)
    movement = _compute_movement_away(player_pos, closest["pos"],
                                       strafe_amount=0.25, strafe_dir=strafe_dir,
                                       walls=walls)
    
    # Fire while kiting back - but only if enemy is hittable (not behind wall)
    has_ammo = bb.get("player.ammo", 0) >= 1
    hittable = closest.get("hittable", True)
    bb.set("decision.movement", movement)
    bb.set("decision.should_attack", has_ammo and hittable)
    bb.set("decision.attack_type", "auto")
    hit_str = "+FIRE" if (has_ammo and hittable) else ""
    bb.set("decision.reason", f"BT: KITE BACK{hit_str} ({int(closest['distance'])}px)")
    return Status.SUCCESS

def act_ranged_attack(bb: Blackboard) -> Status:
    """Keep at optimal range and fire - for snipers/throwers.
    
    Key improvement: properly manage range (back off if too close, 
    approach if too far, strafe at optimal distance).
    """
    player_pos = bb.get("player.pos", (0, 0))
    target = _pick_best_target(bb) or bb.get("enemies_closest")
    if not target:
        return Status.FAILURE

    optimal_range = _get_optimal_range(bb)
    attack_range = bb.get("brawler.attack_range_scaled", 300)
    distance = target["distance"]
    strafe_dir = _update_strafe(bb)
    
    walls = bb.get("map.walls", [])
    # Range management
    range_diff = distance - optimal_range
    
    if range_diff < -optimal_range * 0.2:
        # Too close - kite back with heavy strafe
        movement = _compute_movement_away(player_pos, target["pos"],
                                           strafe_amount=0.4, strafe_dir=strafe_dir,
                                           walls=walls)
    elif range_diff > optimal_range * 0.3:
        # Too far - approach with strafe
        movement = _compute_movement_toward(player_pos, target["pos"],
                                             strafe_amount=0.3, strafe_dir=strafe_dir,
                                             walls=walls)
    else:
        # At good range - pure strafe (perpendicular)
        raw_dx = target["pos"][0] - player_pos[0]
        raw_dy = target["pos"][1] - player_pos[1]
        dx = -raw_dy * strafe_dir
        dy = raw_dx * strafe_dir
        h = 'D' if dx > 0.2 else ('A' if dx < -0.2 else '')
        v = 'S' if dy > 0.2 else ('W' if dy < -0.2 else '')
        movement = (v + h).upper() or "WD"
    
    # Attack if in range and hittable (generous range for reliability)
    # Throwers get extra range (arc trajectory)
    _ps = bb.get("brawler.playstyle", "fighter")
    range_mult = 1.4 if _ps in ("thrower",) else (1.35 if _ps in ("tank", "fighter", "assassin") else 1.25)
    in_fire_range = distance <= attack_range * range_mult
    has_ammo = bb.get("player.ammo", 0) >= 1
    hittable = target.get("hittable", True)
    
    bb.set("decision.movement", movement)
    bb.set("decision.should_attack", in_fire_range and has_ammo and hittable)
    bb.set("decision.attack_type", "auto")
    hittable_str = "HIT" if (hittable and in_fire_range) else "WAIT"
    bb.set("decision.reason", f"BT: RANGED ({int(distance)}px) [{hittable_str}]")
    return Status.SUCCESS

def act_rush_enemy(bb: Blackboard) -> Status:
    """Approach enemy aggressively (tanks, assassins).
    
    Rush with slight diagonal strafe to be harder to hit.
    Attack as soon as in range.
    """
    player_pos = bb.get("player.pos", (0, 0))
    target = _pick_best_target(bb) or bb.get("enemies_closest")
    if not target:
        return Status.FAILURE

    distance = target["distance"]
    attack_range = bb.get("brawler.attack_range_scaled", 300)
    strafe_dir = _update_strafe(bb)
    flank_dir = _update_flank_dir(bb)
    
    walls = bb.get("map.walls", [])
    # More strafe when closer (to dodge), less when far (to close gap fast)
    strafe_frac = min(0.35, max(0.05, 1.0 - distance / max(1, attack_range * 1.5)))
    use_flank = (
        target.get("hittable", True)
        and distance > attack_range * 1.15
        and distance < attack_range * 2.6
    )
    move_target = _compute_flank_target(player_pos, target["pos"], attack_range, flank_dir) if use_flank else target["pos"]
    movement = _compute_movement_toward(player_pos, move_target,
                                         strafe_amount=strafe_frac, strafe_dir=strafe_dir, walls=walls)
    
    # Attack when in range - be generous for melee (auto-aim has its own range check)
    in_range = distance <= attack_range * 1.3  # 30% extra tolerance for melee rush
    has_ammo = bb.get("player.ammo", 0) >= 1
    hittable = target.get("hittable", True)
    
    bb.set("decision.movement", movement)
    bb.set("decision.should_attack", in_range and has_ammo and hittable)
    bb.set("decision.attack_type", "auto")
    fire_str = "+FIRE" if (in_range and has_ammo and hittable) else ""
    flank_tag = " FLANK" if use_flank else ""
    bb.set("decision.reason", f"BT: RUSH{flank_tag}{fire_str} ({int(distance)}px)")
    return Status.SUCCESS

def act_approach_enemy(bb: Blackboard) -> Status:
    """Close distance to out-of-range enemy.
    
    Move with diagonal juke pattern to avoid being easy target.
    Fire when entering attack range to not waste approach time.
    """
    player_pos = bb.get("player.pos", (0, 0))
    target = _pick_best_target(bb) or bb.get("enemies_closest")
    if not target:
        return Status.FAILURE

    distance = target["distance"]
    attack_range = bb.get("brawler.attack_range_scaled", 300)
    strafe_dir = _update_strafe(bb)
    flank_dir = _update_flank_dir(bb)
    
    walls = bb.get("map.walls", [])
    # Light strafe while approaching (15%), use flank route for longer, less predictable pathing.
    use_flank = distance > attack_range * 1.2 and distance < attack_range * 3.0
    move_target = _compute_flank_target(player_pos, target["pos"], attack_range, flank_dir) if use_flank else target["pos"]
    movement = _compute_movement_toward(player_pos, move_target,
                                         strafe_amount=0.15, strafe_dir=strafe_dir, walls=walls)
    
    # Attack if we've entered range while approaching!
    # Use generous threshold - better to fire slightly early than never
    playstyle = bb.get("brawler.playstyle", "fighter")
    # Wider fire range for approach - shoot as soon as possible!
    # Throwers get extra range (arc), melee tighter
    range_mult = 1.2 if playstyle in ("tank", "assassin") else (1.35 if playstyle in ("thrower",) else 1.3)
    in_range = distance <= attack_range * range_mult
    ammo = bb.get("player.ammo", 0)
    hittable = target.get("hittable", True)
    # Fire during approach - don't hold back, every hit counts!
    # Always fire with ANY ammo when in range - don't save ammo while approaching!
    if in_range and hittable and ammo >= 1:
        should_fire = True  # Fire immediately - approaching without shooting wastes time!
    else:
        should_fire = False
    
    bb.set("decision.movement", movement)
    bb.set("decision.should_attack", should_fire)
    if should_fire:
        bb.set("decision.attack_type", "auto")
    if should_fire:
        tag = "APPROACH FLANK+FIRE" if use_flank else "APPROACH+FIRE"
    else:
        tag = "APPROACH FLANK" if use_flank else "APPROACH"
    bb.set("decision.reason", f"BT: {tag} ({int(distance)}px)")
    return Status.SUCCESS

def act_optimal_range_combat(bb: Blackboard) -> Status:
    """Main combat action: maintain optimal range and fire.
    
    This is the core fighting logic for ALL playstyles.
    Manages range, strafes, and fires when appropriate.
    """
    player_pos = bb.get("player.pos", (0, 0))
    target = _pick_best_target(bb) or bb.get("enemies_closest")
    if not target:
        return Status.FAILURE
    
    distance = target["distance"]
    optimal_range = _get_optimal_range(bb)
    attack_range = bb.get("brawler.attack_range_scaled", 300)
    playstyle = bb.get("brawler.playstyle", "fighter")
    strafe_dir = _update_strafe(bb)
    flank_dir = _update_flank_dir(bb)
    
    # Smart ammo management: conserve last ammo for ranged brawlers
    ammo = bb.get("player.ammo", 0)
    max_ammo = bb.get("player.max_ammo", 3)
    enemy_hp = target.get("hp", 100)
    
    # === RANGE MANAGEMENT ===
    range_diff = distance - optimal_range
    tolerance = optimal_range * 0.2

    # Distance-zone hysteresis to prevent CLOSE/MID/FAR thrashing.
    now = float(bb.get("match.time", 0.0) or 0.0)
    if now <= 0:
        now = time.time()
    zone_hold_until = float(bb.get("_combat_zone_hold_until", 0.0) or 0.0)
    current_zone = str(bb.get("_combat_zone", "mid") or "mid")
    close_thr = optimal_range * 0.85
    far_thr = optimal_range * 1.18
    if distance < close_thr:
        desired_zone = "close"
    elif distance > far_thr:
        desired_zone = "far"
    else:
        desired_zone = "mid"
    if now >= zone_hold_until and desired_zone != current_zone:
        bb.set("_combat_zone", desired_zone)
        bb.set("_combat_zone_hold_until", now + 0.25)
        current_zone = desired_zone
    
    walls = bb.get("map.walls", [])
    
    # aMMO-EMPTY KITE: When out of ammo and close, retreat hard
    # Minimal strafe, maximum backward movement to create space while
    # waiting for ammo to reload.  Only kite when within attack range.
    ammo_empty_kite = (ammo == 0 and distance < attack_range * 0.8)
    
    if ammo_empty_kite:
        # Hard retreat - almost no strafe, pure backward movement
        strafe_amt = 0.10  # very low strafe so we actually move AWAY
        movement = _compute_movement_away(player_pos, target["pos"],
                                           strafe_amount=strafe_amt, strafe_dir=strafe_dir,
                                           walls=walls)
    elif current_zone == "close":
        # Too close - back off while strafing (orbit-like, compact amplitude)
        strafe_amt = 0.45 if playstyle in ("sniper", "thrower") else 0.28
        movement = _compute_movement_away(player_pos, target["pos"],
                                           strafe_amount=strafe_amt, strafe_dir=strafe_dir,
                                           walls=walls)
    elif current_zone == "far":
        # Too far - approach oblique and occasionally flank at longer ranges.
        strafe_amt = 0.24
        use_flank = distance > attack_range * 1.35 and distance < attack_range * 3.1
        move_target = _compute_flank_target(player_pos, target["pos"], attack_range, flank_dir) if use_flank else target["pos"]
        movement = _compute_movement_toward(player_pos, move_target,
                                             strafe_amount=strafe_amt, strafe_dir=strafe_dir,
                                             walls=walls)
    else:
        # Mid range duel - stable perpendicular strafe.
        raw_dx = target["pos"][0] - player_pos[0]
        raw_dy = target["pos"][1] - player_pos[1]
        dx = -raw_dy * strafe_dir
        dy = raw_dx * strafe_dir
        h = 'D' if dx > 0.2 else ('A' if dx < -0.2 else '')
        v = 'S' if dy > 0.2 else ('W' if dy < -0.2 else '')
        movement = (v + h).upper() or random.choice(["WA", "WD"])
    
    # === ATTACK DECISION ===
    # Use generous range - better to fire slightly early than never
    # Throwers get extra range (arc trajectory), melee tighter
    range_mult = 1.2 if playstyle in ("tank", "assassin") else (1.4 if playstyle in ("thrower",) else 1.35)
    in_range = distance <= attack_range * range_mult
    has_ammo = bb.get("player.ammo", 0) >= 1
    hittable = target.get("hittable", True)
    
    should_fire = False
    # Phantom enemies = memory-based positions (1.5s stale). Only suppress
    # fire for phantoms FAR beyond attack range where they probably moved.
    is_phantom = target.get("phantom", False)
    phantom_limit = attack_range * 1.4  # Generous: phantoms are recent, still worth shooting (was 1.1)
    if is_phantom and distance > phantom_limit:
        should_fire = False  # Don't waste ammo on ghosts far away
    elif in_range and has_ammo and hittable:
        # ALWAYS fire when in range with ammo - no ammo hoarding!
        # Ammo regenerates fast, missing shots from not firing is worse
        # than wasting the occasional last ammo.
        if ammo_empty_kite and ammo <= 1:
            should_fire = False  # Just got 1 ammo back while kiting - brief save
        else:
            should_fire = True  # Fire with any ammo count!
    
    bb.set("decision.movement", movement)
    bb.set("decision.should_attack", should_fire)
    bb.set("decision.attack_type", "auto")
    
    # Detailed reason
    range_str = "CLOSE" if range_diff < -tolerance else ("FAR" if range_diff > tolerance else "GOOD")
    kite_str = " KITE" if ammo_empty_kite else ""
    hit_str = "HIT" if should_fire else "WAIT"
    phantom_str = " PHANTOM" if is_phantom else ""
    atk_r_str = f" atk_r={int(attack_range)}" if not should_fire else ""
    bb.set("decision.reason", 
           f"BT: COMBAT {playstyle.upper()} ({int(distance)}px range={range_str}{phantom_str}{kite_str}{atk_r_str}) [{hit_str}]")
    return Status.SUCCESS

def act_strafe_and_fire(bb: Blackboard) -> Status:
    """Fallback combat: strafe and fire. Used when no other action applies."""
    return act_optimal_range_combat(bb)

def act_shoot_bush(bb: Blackboard) -> Status:
    """Shoot at the bush where an enemy likely disappeared.
    
    Uses last-known enemy position or nearest dangerous bush.
    Approaches while shooting to flush them out.
    Much more conservative to avoid wasting ammo.
    """
    player_pos = bb.get("player.pos", (0, 0))
    attack_range = bb.get("brawler.attack_range_scaled", 300)
    
    # Cooldown: don't spam bush checks too fast (3s between shots)
    last_bush_shot = bb.get("_last_bush_shot_time", 0)
    now = bb.get("match.time", 0)
    if now - last_bush_shot < 3.0:
        return Status.FAILURE  # Cooldown active, let patrol happen instead
    
    # Pick target: prefer dangerous bush, fall back to last-known pos
    target_pos = None
    dangerous = bb.get("map.dangerous_bushes", [])
    
    if dangerous:
        # Find closest dangerous bush in attack range
        best = None
        best_dist = float('inf')
        for bush in dangerous:
            d = math.hypot(bush["pos"][0] - player_pos[0],
                           (bush["pos"][1] - player_pos[1]) * 1.25)
            if d < best_dist:
                best_dist = d
                best = bush
        if best:
            target_pos = best["pos"]
    
    if not target_pos:
        target_pos = bb.get("enemy.last_known_pos")
    
    if not target_pos:
        return Status.FAILURE
    
    distance = math.hypot(target_pos[0] - player_pos[0],
                          (target_pos[1] - player_pos[1]) * 1.25)
    
    # Move toward the bush (approach only, don't shoot from far)
    strafe_dir = _update_strafe(bb)
    walls = bb.get("map.walls", [])
    movement = _compute_movement_toward(player_pos, target_pos,
                                         strafe_amount=0.1, strafe_dir=strafe_dir,
                                         walls=walls)
    
    # Only shoot if close AND have full ammo (1 shot max, keep 2 for combat)
    in_range = distance <= attack_range * 1.0  # Tighter range (was 1.2)
    should_fire = in_range and bb.get("player.ammo", 0) >= 3  # Need full ammo
    
    if should_fire:
        bb.set("_last_bush_shot_time", now)
        bb.set("decision.bush_target_pos", target_pos)
    
    bb.set("decision.movement", movement)
    bb.set("decision.should_attack", should_fire)
    bb.set("decision.attack_type", "aimed")
    tag = "BUSH CHECK" if should_fire else "APPROACH BUSH"
    bb.set("decision.reason", f"BT: {tag} ({int(distance)}px)")
    return Status.SUCCESS


def act_disengage_heal(bb: Blackboard) -> Status:
    """Disengage from combat to let HP regenerate.
    
    - Move away from enemy with strong strafe
    - Do NOT fire (firing resets regen timer)
    - Duration: 2-3 seconds
    """
    player_pos = bb.get("player.pos", (0, 0))
    closest = bb.get("enemies_closest")
    walls = bb.get("map.walls", [])
    now = bb.get("match.time", 0)
    strafe_dir = _update_strafe(bb)
    
    # Set disengage end time if not set yet
    disengage_end = bb.get("_disengage_end_time", 0)
    if now > disengage_end:
        duration = random.uniform(2.0, 3.0)
        bb.set("_disengage_end_time", now + duration)
        bb.set("_last_disengage_end", now + duration)
    
    if closest:
        movement = _compute_movement_away(player_pos, closest["pos"],
                                           strafe_amount=0.4, strafe_dir=strafe_dir,
                                           walls=walls)
    else:
        movement = random.choice(["SA", "SD", "S"])
    
    bb.set("decision.movement", movement)
    bb.set("decision.should_attack", False)  # DON'T FIRE - regen resets!
    hp = bb.get("player.hp", 0)
    bb.set("decision.reason", f"BT: HEAL ({hp}% -> regen)")
    return Status.SUCCESS


def act_reactive_dodge(bb: Blackboard) -> Status:
    """Briefly dodge perpendicular to the attacker after taking damage.
    
    Duration: 0.3s - then BT resumes normal combat.
    """
    player_pos = bb.get("player.pos", (0, 0))
    closest = bb.get("enemies_closest")
    walls = bb.get("map.walls", [])
    now = bb.get("match.time", 0)
    
    # Set dodge duration (0.3s)
    bb.set("_reactive_dodge_until", now + 0.3)
    
    if closest:
        enemy_pos = closest["pos"]
        # Pure perpendicular movement (100% strafe, 0% toward/away)
        dx = enemy_pos[0] - player_pos[0]
        dy = enemy_pos[1] - player_pos[1]
        mag = math.hypot(dx, dy)
        if mag > 1:
            # Pick random perpendicular direction
            side = 1 if random.random() < 0.5 else -1
            perp_x = -dy / mag * side
            perp_y = dx / mag * side
            h = 'D' if perp_x > 0.2 else ('A' if perp_x < -0.2 else '')
            v = 'S' if perp_y > 0.2 else ('W' if perp_y < -0.2 else '')
            movement = (v + h).upper() or "WA"
        else:
            movement = random.choice(["WA", "WD"])
        # Wall check
        movement = _wall_adjust_movement(player_pos, movement, walls)
    else:
        movement = random.choice(["WA", "WD", "SA", "SD"])
    
    # Only fire if we can actually see an enemy to aim at
    # Blind fire with enemies=0 just wastes ammo (auto-aim has no target)
    has_ammo = bb.get("player.ammo", 0) >= 1
    has_target = closest is not None
    should_fire = has_ammo and has_target
    
    bb.set("decision.movement", movement)
    bb.set("decision.should_attack", should_fire)
    bb.set("decision.reason", f"BT: DODGE HIT! +FIRE" if should_fire else "BT: DODGE HIT!")
    return Status.SUCCESS


def act_idle_strafe(bb: Blackboard) -> Status:
    """Fallback: move toward last-known enemy or push forward (spawn-aware)."""
    player_pos = bb.get("player.pos", (0, 0))
    last_known = bb.get("enemy.last_known_pos")
    walls = bb.get("map.walls", [])
    if last_known:
        dist = math.hypot(last_known[0] - player_pos[0], last_known[1] - player_pos[1])
        if dist > 60:
            movement = _compute_movement_toward(player_pos, last_known,
                                                strafe_amount=0.1, walls=walls)
            bb.set("decision.movement", movement)
            bb.set("decision.reason", "BT: HUNT")
            return Status.SUCCESS
    # Push forward (spawn-aware) using pathfinder
    spawn_side = bb.get("map.spawn_side", None)
    if spawn_side == "left":
        fwd_target = (min(1840, player_pos[0] + 220), player_pos[1])
    elif spawn_side == "right":
        fwd_target = (max(80, player_pos[0] - 220), player_pos[1])
    elif spawn_side == "top":
        fwd_target = (player_pos[0], min(990, player_pos[1] + 220))
    else:  # bottom or unknown
        fwd_target = (player_pos[0], max(40, player_pos[1] - 220))
    movement = _compute_movement_toward(player_pos, fwd_target,
                                         strafe_amount=0.05, walls=walls)
    bb.set("decision.movement", movement)
    bb.set("decision.reason", "BT: PUSH FORWARD")
    return Status.SUCCESS


# --- tREE BUILDER ---

def build_combat_tree() -> Selector:
    """Construct the full combat behavior tree.
    
    Priority order:
    1. Emergency (storm/gas)
    2. Reactive dodge (just got hit -> brief perpendicular dodge)
    3. Dead
    4. No enemies -> bush check / patrol
    5. Disengage to heal (mid HP + enemy far) ← ABOVE combat!
    6. Abilities (super/gadget when smart)
    7. Combat:
       a. Low HP -> retreat with counter-fire
       b. Enemy killable -> aggressive finish
       c. Too close (ranged) -> kite back
       d. Respawn shield -> rush
       e. Optimal range -> maintain and fire (main fighting mode)
       f. Enemy far (aggressive) -> approach
       g. Default -> range combat
    """

    # emergency: flee storm/gas
    emergency = Sequence("Emergency", [
        Condition("InStorm?", cond_in_storm),
        Action("FleeStorm", act_flee_storm),
    ])

    # dodge incoming projectiles (ALSO fire while dodging!)
    def _dodge_and_fire(bb: Blackboard) -> Status:
        bb.set("decision.movement", bb.get("dodge.direction", "WA"))
        # Attack while dodging - dodging and shooting are not mutually exclusive!
        closest = bb.get("enemies_closest")
        if closest:
            atk_range = bb.get("brawler.attack_range_scaled", 300)
            ammo = bb.get("player.ammo", 0)
            dist = closest.get("distance", 999)
            if dist <= atk_range * 1.3 and ammo >= 1:
                bb.set("decision.should_attack", True)
                bb.set("decision.attack_type", "auto")
                bb.set("decision.reason", f"BT: DODGE+FIRE ({int(dist)}px)")
                return Status.SUCCESS
        bb.set("decision.reason", "BT: DODGE")
        return Status.SUCCESS

    def _dodge_valid(bb: Blackboard) -> bool:
        """Only dodge if projectile detected AND enemies seen recently (or damage taken)."""
        if not bb.get("dodge.active", False):
            return False
        # Always dodge if enemies are visible right now
        if bb.get("enemies_count", 0) > 0:
            return True
        # Dodge if we took damage recently (might be hidden thrower)
        hp = bb.get("player.hp", 100)
        if hp < 95:
            return True
        # Dodge if enemy was seen within 2s (might've just gone behind wall)
        now = bb.get("match.time", 0)
        last_seen = bb.get("enemy.last_seen_time", 0)
        if now - last_seen < 2.0:
            return True
        # No enemies, full HP, no recent enemy -> probably false positive
        return False

    dodge = Sequence("Dodge", [
        Condition("ProjectileIncoming?", _dodge_valid),
        Action("DodgeProjectile", _dodge_and_fire),
    ])

    # reactive dodge: got hit -> brief perpendicular dodge
    reactive_dodge = Sequence("ReactDodge", [
        Condition("TakingDamage?", cond_taking_damage),
        Action("DodgeHit", act_reactive_dodge),
    ])

    # dead/respawning
    dead = Sequence("Dead", [
        Condition("IsDead?", cond_is_dead),
        Action("WaitDead", act_wait_dead),
    ])

    # no enemies visible: bush check or patrol
    bush_check = Sequence("BushCheck", [
        Condition("EnemyInBush?", cond_enemy_in_bush),
        Action("ShootBush", act_shoot_bush),
    ])

    patrol_only = Action("Patrol", act_patrol)

    no_enemy = Sequence("NoEnemy", [
        Condition("NoEnemies?", cond_no_enemies),
        Selector("BushOrPatrol", [
            bush_check,
            patrol_only,
        ]),
    ])

    # disengage to heal (mid HP, enemy far enough)
    disengage = Sequence("Disengage", [
        Condition("ShouldHeal?", cond_should_disengage_heal),
        Action("Heal", act_disengage_heal),
    ])

    # super usage (when smart)
    use_super = Sequence("UseSuper", [
        Condition("ShouldSuper?", cond_should_use_super),
        Action("Super", act_use_super),
    ])

    # gadget usage
    use_gadget = Sequence("UseGadget", [
        Condition("ShouldGadget?", cond_should_use_gadget),
        Action("Gadget", act_use_gadget),
    ])

    # combat subtree
    retreat = Sequence("Retreat", [
        Condition("LowHP?", cond_low_hp),
        Action("Retreat", act_retreat),
    ])

    finish = Sequence("FinishKill", [
        Condition("EnemyKillable?", cond_enemy_killable),
        Action("FinishKill", act_finish_kill),
    ])

    kite = Sequence("KiteBack", [
        Condition("TooClose?", cond_too_close),
        Action("KiteBack", act_kite_back),
    ])

    ranged = Sequence("RangedAttack", [
        Condition("KeepMaxRange?", cond_keep_max_range),
        Action("Ranged", act_ranged_attack),
    ])

    respawn_rush = Sequence("RespawnRush", [
        Condition("RespawnShield?", cond_respawn_shield),
        Action("Rush", act_rush_enemy),
    ])

    approach = Sequence("Approach", [
        Condition("EnemyFarAggressive?", cond_enemy_far_and_aggressive),
        Action("Approach", act_approach_enemy),
    ])

    default_combat = Action("OptimalCombat", act_optimal_range_combat)

    combat = Selector("Combat", [
        retreat,
        finish,
        kite,
        ranged,
        respawn_rush,
        approach,
        default_combat,
    ])

    abilities = Selector("Abilities", [
        use_super,
        use_gadget,
    ])

    fallback = Action("IdleStrafe", act_idle_strafe)

    combat_with_abilities = Sequence("CombatWithAbilities", [
        ForceSuccess("TryAbilities", abilities),
        combat,
    ])

    # root
    # Priority: Emergency > Dead > Combat (with dodge) > Disengage (heal) > NoEnemy > Fallback
    # Combat remains primary when enemies are visible; disengage is now conditional fallback.
    root = Selector("Root", [
        emergency,
        dead,
        combat_with_abilities,       # Combat when enemies visible
        disengage,                   # Heal only when combat branch cannot engage
        reactive_dodge,              # Dodge when hit but no enemies detected
        dodge,                       # Projectile dodge (lower priority than combat)
        no_enemy,
        fallback,
    ])

    return root


# --- mAIN BT COMBAT CONTROLLER ---

class BTCombat:
    """High-level controller that manages the BT, blackboard, and subsystems.

    Drop-in replacement for the monolithic get_movement() decision logic.
    """

    def __init__(self, play_instance):
        self.play = play_instance
        self.blackboard = Blackboard()
        self.root = build_combat_tree()

        # Optional subsystem references (initialized if available)
        self._enemy_tracker = None
        self._projectile_detector = None
        self._spatial_memory = None
        self._combo_engine = None
        self._opponent_model = None
        self._manual_aimer = None
        self._ammo_reader = None
        self._hp_estimator = None

        # RL Training components
        self._state_encoder = None
        self._reward_calculator = None
        self._rl_trainer = None
        self._last_state = None
        self._last_action = 0
        self._last_log_prob = 0.0
        self._last_value = 0.0
        self._rl_mode = "rules"  # "rules" / "hybrid" / "full_rl"
        self._kpi_adj_bonus_base = 0.05
        self._kpi_adj_bonus_threat_scale = 0.05
        self._kpi_adj_attack_block_base_penalty = 0.02
        self._kpi_adj_attack_block_threat_penalty = 0.04
        self._kpi_adj_pattern_block_penalty = 0.012
        self._kpi_adj_clip_abs = 2.5
        self._kpi_adj_profile = "balanced"

        # Debug/logging
        self._debug_print = False  # DISABLED: kills IPS (7 -> 60+)
        self._log_every_n_ticks = 10  # Log every N ticks when debug enabled

        self._tick_count = 0
        self._last_tree_path = ""
        
        # Attack rate limiting
        self._last_attack_time = 0.0
        self._min_attack_interval = 0.08  # Min 80ms between attacks (was 150ms)

        # Hold-attack (charge) state for brawlers like Hank, Angelo
        self._hold_attack_start = None    # timestamp when hold began (None = not holding)
        self._hold_attack_duration = 0    # base charge time from brawler config
        self._is_hold_brawler = False     # cached flag

        # Enemy memory: persist last-seen positions when detection flickers
        self._enemy_memory = []       # [{"pos", "distance", "hp", "hittable", "last_seen"}]
        self._enemy_memory_duration = 1.5  # seconds to remember enemies after losing sight
        self._last_match_start = 0  # Track round/match changes
        self._last_super_time_bt = 0.0  # BT-level super cooldown
        
        # Kill/Death/Damage tracking for rewards
        self._prev_hp = 100.0
        self._prev_enemy_count = 0
        self._prev_enemy_hp = 100.0  # Closest enemy HP
        self._session_kills = 0
        self._session_deaths = 0
        self._session_damage_dealt = 0.0
        self._session_damage_taken = 0.0
        self._match_kills = 0
        self._match_deaths = 0
        self._last_play_kill_count = 0  # Sync kills from play.py's detection
        self._last_tobs_total_damage = 0.0
        self._last_tobs_total_matches = 0
        # Accumulators for RL - gather events between RL ticks (every 3rd tick)
        self._rl_accum_damage_dealt = 0.0
        self._rl_accum_damage_taken = 0.0
        self._rl_accum_kills = 0
        self._rl_accum_deaths = 0
        self._rl_accum_attacked = False
        self._rl_move_override_applied = 0
        self._rl_move_override_blocked = 0
        self._rl_move_override_blocked_attack_window = 0
        self._rl_move_override_blocked_pattern_pressure = 0
        self._rl_move_override_critical_applied = 0
        self._peek_active_frames = 0
        self._water_pressure_frames = 0
        self._enemy_pattern_pressure_frames = 0
        self._enemy_attack_soon_frames = 0

        # Movement stabilization (prevents rapid direction thrashing)
        # 0.50s commit = direction held for ~8-15 frames before switching.
        # Emergency bypass still allows instant changes for dodge/retreat.
        self._movement_commit_min_s = 0.50
        self._last_committed_movement = ""
        self._last_movement_switch_time = 0.0

        self._init_subsystems()

    def _apply_enemy_memory(self, bb: Blackboard):
        """Keep enemy positions alive for 1.5s when detection flickers.

        Without this, the bot loses all enemy info on frames where the
        ONNX model fails to detect enemies, causing constant flip-flop
        between Combat and NoEnemy branches.
        """
        now = time.time()
        current_enemies = bb.get("enemies", [])

        if current_enemies:
            # Detection found enemies -> update memory with fresh data
            self._enemy_memory = [{
                "pos": e["pos"],
                "distance": e["distance"],
                "bbox": e.get("bbox"),
                "hp": e.get("hp", -1),
                "hittable": e.get("hittable", True),
                "last_seen": now,
            } for e in current_enemies]
        elif self._enemy_memory:
            # No detection this frame - inject memory if still fresh
            fresh = [m for m in self._enemy_memory
                     if now - m["last_seen"] < self._enemy_memory_duration]
            if fresh:
                player_pos = bb.get("player.pos", (0, 0))
                phantom_enemies = []
                for m in fresh:
                    # Recalculate distance from current player pos
                    dx = m["pos"][0] - player_pos[0]
                    dy = m["pos"][1] - player_pos[1]
                    dist = math.hypot(dx, dy)
                    phantom_enemies.append({
                        "pos": m["pos"],
                        "distance": dist,
                        "bbox": m.get("bbox"),
                        "hp": m.get("hp", -1),
                        "hittable": True,
                        "phantom": True,
                    })
                phantom_enemies.sort(key=lambda e: e["distance"])
                bb["enemies"] = phantom_enemies
                bb["enemies_count"] = len(phantom_enemies)
                bb["enemies_closest"] = phantom_enemies[0]
            else:
                # Memory expired - clear
                self._enemy_memory = []

    @staticmethod
    def _normalize_movement_keys(movement: str) -> str:
        """Normalize arbitrary WASD string to canonical movement key format."""
        if not movement:
            return ""
        keys = [ch for ch in str(movement).upper() if ch in "WASD"]
        if not keys:
            return ""

        has_w = "W" in keys
        has_s = "S" in keys
        has_a = "A" in keys
        has_d = "D" in keys

        v = ""
        if has_w and not has_s:
            v = "W"
        elif has_s and not has_w:
            v = "S"

        h = ""
        if has_a and not has_d:
            h = "A"
        elif has_d and not has_a:
            h = "D"

        return (v + h).upper()

    def _stabilize_movement(self, bb: Blackboard, movement: str) -> str:
        """Apply short movement commitment to avoid jitter between directions."""
        movement = self._normalize_movement_keys(movement)
        now = time.time()

        if movement == self._last_committed_movement:
            return movement

        hp = float(bb.get("player.hp", 100) or 100)
        retreat_hp = float(bb.get("style.hp_retreat", 45) or 45)
        closest = bb.get("enemies_closest") or {}
        dist = float(closest.get("distance", 999.0) or 999.0)
        attack_range = float(bb.get("brawler.attack_range_scaled", 300) or 300)
        emergency = (
            bool(bb.get("dodge.active", False))
            or _storm_threat_active(bb)
            or hp <= (retreat_hp + 3)
            or dist <= attack_range * 0.55
        )

        if (not emergency
                and self._last_committed_movement
                and (now - self._last_movement_switch_time) < self._movement_commit_min_s):
            return self._last_committed_movement

        self._last_committed_movement = movement
        self._last_movement_switch_time = now
        return movement

    def _init_subsystems(self):
        """Try to initialize optional subsystems (graceful if imports fail)."""
        try:
            from enemy_tracker import EnemyTracker
            self._enemy_tracker = EnemyTracker()
        except ImportError:
            pass

        try:
            from projectile_detector import ProjectileDetector
            self._projectile_detector = ProjectileDetector()
        except ImportError:
            pass

        try:
            from spatial_memory import SpatialMemory
            wc = getattr(self.play, 'window_controller', None) if self.play else None
            w = getattr(wc, 'width', 1920) or 1920 if wc else 1920
            h = getattr(wc, 'height', 1080) or 1080 if wc else 1080
            self._spatial_memory = SpatialMemory(w, h)
        except (ImportError, Exception):
            pass

        # Use play.py's pathfinder if available; otherwise create our own
        try:
            from pathfinder import PathPlanner
            if self.play and hasattr(self.play, '_path_planner') and self.play._path_planner is not None:
                self._path_planner = self.play._path_planner
            else:
                self._path_planner = PathPlanner(cell_size=40)
            # Also share spatial memory with play.py if it has one
            if self.play and hasattr(self.play, '_spatial_memory') and self.play._spatial_memory is not None:
                self._spatial_memory = self.play._spatial_memory
        except (ImportError, Exception):
            self._path_planner = None

        try:
            from combo_engine import ComboEngine
            self._combo_engine = ComboEngine()
        except ImportError:
            pass

        try:
            from rl.opponent_model import OpponentModel
            self._opponent_model = OpponentModel()
        except ImportError:
            pass

        try:
            from manual_aim import ManualAimer
            wc = getattr(self.play, 'window_controller', None) if self.play else None
            self._manual_aimer = ManualAimer(wc)
        except (ImportError, Exception):
            pass

        try:
            from visual_ammo_reader import AmmoReader
            wc = getattr(self.play, 'window_controller', None) if self.play else None
            self._ammo_reader = AmmoReader(wc)
        except (ImportError, Exception):
            pass

        try:
            from hp_estimator import HPEstimator
            self._hp_estimator = HPEstimator()
        except ImportError:
            pass

        # RL Components (StateEncoder, RewardCalculator, PPOTrainer)
        try:
            from rl.state_encoder import StateEncoder
            self._state_encoder = StateEncoder()
        except ImportError:
            pass

        try:
            from rl.reward_calculator import RewardCalculator
            self._reward_calculator = RewardCalculator()
        except ImportError:
            pass

        # Load RL config
        try:
            from utils import load_toml_as_dict
            gc = load_toml_as_dict("cfg/general_config.toml")
            self._rl_mode = str(gc.get("ai_mode", "rules")).lower()
            rl_training = str(gc.get("rl_training_enabled", "no")).lower() in ("yes", "true", "1")

            self._kpi_adj_profile = str(gc.get("rl_kpi_adj_profile", "balanced") or "balanced").lower()
            kpi_profiles = {
                "conservative": {
                    "bonus_base": 0.035,
                    "bonus_threat_scale": 0.035,
                    "attack_block_base_penalty": 0.03,
                    "attack_block_threat_penalty": 0.055,
                    "pattern_block_penalty": 0.016,
                    "clip_abs": 2.2,
                },
                "balanced": {
                    "bonus_base": 0.05,
                    "bonus_threat_scale": 0.05,
                    "attack_block_base_penalty": 0.02,
                    "attack_block_threat_penalty": 0.04,
                    "pattern_block_penalty": 0.012,
                    "clip_abs": 2.5,
                },
                "aggressive": {
                    "bonus_base": 0.065,
                    "bonus_threat_scale": 0.07,
                    "attack_block_base_penalty": 0.014,
                    "attack_block_threat_penalty": 0.028,
                    "pattern_block_penalty": 0.009,
                    "clip_abs": 3.0,
                },
            }
            if self._kpi_adj_profile in kpi_profiles:
                profile = kpi_profiles[self._kpi_adj_profile]
                self._kpi_adj_bonus_base = float(profile["bonus_base"])
                self._kpi_adj_bonus_threat_scale = float(profile["bonus_threat_scale"])
                self._kpi_adj_attack_block_base_penalty = float(profile["attack_block_base_penalty"])
                self._kpi_adj_attack_block_threat_penalty = float(profile["attack_block_threat_penalty"])
                self._kpi_adj_pattern_block_penalty = float(profile["pattern_block_penalty"])
                self._kpi_adj_clip_abs = float(profile["clip_abs"])

            def _cfg_float(key: str, default: float) -> float:
                try:
                    return float(gc.get(key, default) or default)
                except Exception:
                    return default

            self._kpi_adj_bonus_base = _cfg_float("rl_kpi_adj_bonus_base", self._kpi_adj_bonus_base)
            self._kpi_adj_bonus_threat_scale = _cfg_float(
                "rl_kpi_adj_bonus_threat_scale", self._kpi_adj_bonus_threat_scale
            )
            self._kpi_adj_attack_block_base_penalty = _cfg_float(
                "rl_kpi_adj_attack_block_base_penalty", self._kpi_adj_attack_block_base_penalty
            )
            self._kpi_adj_attack_block_threat_penalty = _cfg_float(
                "rl_kpi_adj_attack_block_threat_penalty", self._kpi_adj_attack_block_threat_penalty
            )
            self._kpi_adj_pattern_block_penalty = _cfg_float(
                "rl_kpi_adj_pattern_block_penalty", self._kpi_adj_pattern_block_penalty
            )
            self._kpi_adj_clip_abs = max(0.1, _cfg_float("rl_kpi_adj_clip_abs", self._kpi_adj_clip_abs))

            if rl_training and self._state_encoder:
                from rl.trainer import PPOTrainer
                state_dim = 89
                self._rl_trainer = PPOTrainer(
                    state_dim=state_dim,
                    batch_size=64,
                    entropy_coef=0.03,
                    lr=3e-4,
                )
                # Try to load existing model
                self._rl_trainer.load()
                print(f"[RL] Trainer initialized (state_dim={state_dim})")
        except Exception as e:
            print(f"[RL] Trainer init failed: {e}")

        # Print subsystem summary
        subs = [
            ("EnemyTracker", self._enemy_tracker),
            ("ProjectileDetector", self._projectile_detector),
            ("SpatialMemory", self._spatial_memory),
            ("ComboEngine", self._combo_engine),
            ("OpponentModel", self._opponent_model),
            ("ManualAimer", self._manual_aimer),
            ("AmmoReader", self._ammo_reader),
            ("HPEstimator", self._hp_estimator),
            ("StateEncoder", self._state_encoder),
            ("RewardCalculator", self._reward_calculator),
        ]
        active = sum(1 for _, v in subs if v is not None)
        print(f"[BT] Initialized: {active}/{len(subs)} subsystems active")

    def tick(self, data: dict, frame, brawler: str) -> str:
        """Run one BT tick and return movement keys (e.g. 'WD', 'SA').

        Also executes attack actions if decided by the tree.
        """
        bb = self.blackboard
        self._tick_count += 1

        # 0. Detect round/match start -> clear stale data
        # If play instance just reset (death_count jumped or match restarted)
        if self.play:
            match_start = getattr(self.play, '_match_start_time', 0)
            prev_match_start = getattr(self, '_last_match_start', 0)
            if match_start != prev_match_start and match_start > 0:
                # New round/match started! Clear all stale combat data
                self._enemy_memory = []
                bb.set("enemy.last_known_pos", None)
                bb.set("enemy.last_seen_time", 0)
                bb.set("match.start_time", time.time())
                bb.set("_last_bush_shot_time", 0)
                bb.set("_reactive_dodge_until", 0)
                self._last_match_start = match_start
                if self._debug_print:
                    print("[BT] Round/match start - cleared stale data")

        # 1. Populate blackboard
        populate_blackboard(bb, self.play, data, frame, brawler)

        if bb.get("player.peek_phase", "idle") != "idle":
            self._peek_active_frames += 1
        # Count water pressure only when combat-relevant (enemy visible),
        # otherwise this metric gets inflated on water-heavy maps.
        if bb.get("map.water_nearby", 0) > 0 and int(bb.get("enemies_count", 0) or 0) > 0:
            self._water_pressure_frames += 1

        # 1a. Enemy memory: persist last-known positions
        self._apply_enemy_memory(bb)

        # 1b. Cancel pending hold-attack if no enemies are visible or player died
        if self._hold_attack_start is not None and self._is_hold_brawler:
            enemies_visible = int(bb.get("enemies_count", 0) or 0) > 0
            player_alive = bb.get("player.hp", 100) > 0
            if not enemies_visible or not player_alive:
                # Release the held button immediately (cancel charge)
                if self.play:
                    self.play.attack(touch_up=True, touch_down=False)
                self._hold_attack_start = None
                if self._debug_print:
                    print("[BT] HOLD-ATTACK CANCELLED: no enemies / dead")

        # 1c. Detect kills, deaths, damage events
        current_hp = bb.get("player.hp", 100)
        current_enemy_count = bb.get("enemies_count", 0)
        closest = bb.get("enemies_closest")
        current_enemy_hp = closest.get("hp", 100) if closest else 100
        
        # Player died this frame? Use play.py's respawn-based death detection
        # which is more reliable than HP=0 (HP bar may not drop to 0 visually)
        player_just_died = False
        if self.play:
            play_deaths = getattr(self.play, '_death_count', 0)
            if play_deaths > self._match_deaths:
                player_just_died = True
                new_deaths = play_deaths - self._match_deaths
                self._session_deaths += new_deaths
                self._match_deaths = play_deaths
        else:
            # Fallback: HP-based detection
            player_just_died = (current_hp <= 0 and self._prev_hp > 0)
            if player_just_died:
                self._session_deaths += 1
                self._match_deaths += 1
        bb.set("player.just_died", player_just_died)
        bb.set("player.prev_hp", self._prev_hp)
        
        # Damage taken - require at least 3% HP drop to filter flicker
        hp_drop = max(0, self._prev_hp - current_hp)
        if hp_drop >= 3:  # 3% HP drop minimum (was 10, too strict)
            damage_taken = hp_drop * 32
        else:
            damage_taken = 0  # Ignore minor HP reading fluctuations
        # Fallback for death frames where HP estimator misses the drop:
        # if we detected a new death but no HP delta, infer at least the
        # remaining HP that had to be lost to die.
        if player_just_died and damage_taken == 0 and self._prev_hp > 0:
            damage_taken = max(damage_taken, self._prev_hp * 32)
        if damage_taken > 0:
            self._session_damage_taken += damage_taken
        bb.set("player.damage_taken", damage_taken)
        
        # Enemy killed? Use play.py's more reliable kill detection which
        # tracks HP-before-disappear with generous thresholds (50% if recently
        # attacked, 35% otherwise) instead of our strict <30% check that
        # almost never fires due to HP smoothing lag.
        enemy_just_died = False
        if self.play:
            play_kills = getattr(self.play, '_enemies_killed_this_match', 0)
            new_kills = play_kills - self._last_play_kill_count
            if new_kills > 0:
                enemy_just_died = True
                self._session_kills += new_kills
                self._match_kills += new_kills
                self._last_play_kill_count = play_kills
        bb.set("enemy.just_died", enemy_just_died)
        bb.set("enemy.prev_hp", self._prev_enemy_hp)
        
        # Damage dealt to enemy - compute when enemies are visible.
        # Allow brief detection gaps (<0.5 s): if enemies vanished for only
        # a few frames but reappeared, use the stored _prev_enemy_hp to
        # bridge the gap instead of throwing away the HP delta.
        damage_dealt = 0
        if current_enemy_count > 0 and current_enemy_hp >= 0:
            if self._prev_enemy_count > 0 or (
                self._prev_enemy_count == 0
                and hasattr(self, '_last_enemy_hp_time')
                and (time.time() - self._last_enemy_hp_time) < 0.5
            ):
                raw_dealt = max(0, self._prev_enemy_hp - current_enemy_hp)
                if raw_dealt >= 3:  # Filter tiny HP flicker
                    damage_dealt = raw_dealt * 32
            self._last_enemy_hp_time = time.time()
        if damage_dealt > 0:
            self._session_damage_dealt += damage_dealt
        bb.set("player.damage_dealt", damage_dealt)
        bb.set("enemy.hp", current_enemy_hp)
        
        # Accumulate events for RL (consumed every 3rd tick by _rl_step)
        self._rl_accum_damage_dealt += damage_dealt
        self._rl_accum_damage_taken += damage_taken
        if enemy_just_died:
            self._rl_accum_kills += 1
        if player_just_died:
            self._rl_accum_deaths += 1

        # Store totals for overlay/stats
        bb.set("stats.kills", self._match_kills)
        bb.set("stats.deaths", self._match_deaths)
        bb.set("stats.damage_dealt", self._session_damage_dealt)
        bb.set("stats.damage_taken", self._session_damage_taken)
        
        # Apply adaptive aggression based on performance
        adaptive_aggr = self._get_adaptive_aggression()
        bb.set("aggression", adaptive_aggr)
        
        # Adjust retreat threshold based on performance
        # More kills = lower retreat threshold (stay aggressive)
        # More deaths = higher retreat threshold (play safe)
        base_retreat = bb.get("style.hp_retreat", 45)
        adjusted_retreat = base_retreat * (2.0 - adaptive_aggr)  # Inverse: aggressive = lower retreat
        bb.set("style.hp_retreat", max(20, min(45, adjusted_retreat)))
        
        # Adjust attack interval based on performance  
        # Aggressive = faster attacks, cautious = slower/more careful
        base_interval = bb.get("style.attack_interval", 0.15)
        adjusted_interval = base_interval / adaptive_aggr  # Aggressive = faster
        # Cap at 0.20 max - NEVER let deaths slow attack speed too much!
        self._min_attack_interval = max(0.06, min(0.20, adjusted_interval))
        
        # Update previous-frame state
        self._prev_hp = current_hp
        self._prev_enemy_count = current_enemy_count
        # Only update prev_enemy_hp when enemies are visible; when they vanish,
        # keep the last real HP reading (not 100) so kill detection and rewards
        # can use the actual last-known value.
        if current_enemy_count > 0 and current_enemy_hp >= 0:
            self._prev_enemy_hp = current_enemy_hp

        # 2. Update subsystems
        self._update_subsystems(data, frame, brawler)

        if float(bb.get("enemy.pattern_pressure", 0.0) or 0.0) > 0.55:
            self._enemy_pattern_pressure_frames += 1
        if float(bb.get("enemy.predicted_attack_soon", 0.0) or 0.0) > 0.5:
            self._enemy_attack_soon_frames += 1

        # 2b. Update ammo regeneration (was only in legacy get_movement!)
        if self.play:
            import time as _time
            self.play._update_ammo(_time.time())

        # 3. Reset decision keys from previous tick
        bb.set("decision.movement", "")
        bb.set("decision.should_attack", False)
        bb.set("decision.attack_type", "auto")
        bb.set("decision.reason", "")
        bb.set("decision.use_super", False)
        bb.set("decision.use_gadget", False)
        bb.set("decision.bush_target_pos", None)

        # 3b. Set module-level pathfinder for _compute_movement funcs
        global _active_pathfinder, _active_spatial_memory, _active_storm_center
        _active_pathfinder = getattr(self, '_path_planner', None)
        _active_spatial_memory = self._spatial_memory
        # Set storm center for flee logic - helps concavity escape toward safe zone
        if self.play:
            _storm_delay_over = True
            try:
                _storm_delay_over = bool(self.play._is_storm_flee_delay_over())
            except Exception:
                _storm_delay_over = True
            if _storm_delay_over and (getattr(self.play, '_gas_active', False) or getattr(self.play, '_in_storm', False)):
                _active_storm_center = getattr(self.play, '_storm_center', None)
            else:
                _active_storm_center = None
        else:
            _active_storm_center = None

        # 4. Tick the behavior tree
        self.root.tick(bb)

        # 5. Read decisions from blackboard
        movement = self._normalize_movement_keys(bb.get("decision.movement", ""))
        should_attack = bb.get("decision.should_attack", False)
        attack_type = bb.get("decision.attack_type", "auto")
        reason = bb.get("decision.reason", "")

        # Global HP uncertainty safety gate (applies to all behavior branches)
        if should_attack:
            block_attack, block_reason = _should_block_attack_for_hp_safety(bb)
            if block_attack:
                should_attack = False
                bb.set("decision.should_attack", False)
                reason = (reason + " | " if reason else "") + f"BT: FIRE BLOCK ({block_reason})"

        # 6. Execute abilities (super/gadget) with aim
        use_super = bb.get("decision.use_super", False)
        use_gadget = bb.get("decision.use_gadget", False)
        
        now = time.time()
        if use_super and self.play:
            try:
                self._execute_super(bb)
            except Exception:
                pass
        elif use_gadget and self.play:
            try:
                self.play.window_controller.press_key("G")
                print(f"[BT] GADGET USED!")
            except Exception:
                pass

        # 7. Execute attack if decided (with rate limiting)
        if should_attack and (now - self._last_attack_time) >= self._min_attack_interval:
            try:
                self._execute_attack(bb, attack_type)
                self._last_attack_time = now
            except Exception as e:
                # Never let transient input/ADB errors kill the combat loop.
                try:
                    if self.play:
                        self.play.attack(touch_up=True, touch_down=False)
                except Exception:
                    pass
                reason = (reason + " | " if reason else "") + f"BT: ATTACK ERROR ({type(e).__name__})"

        # 8. Update play instance with reason
        if self.play:
            self.play.last_decision_reason = reason
        self._last_tree_path = get_active_path(self.root)

        # 9. Combo engine tick
        if self._combo_engine and self._combo_engine._active_combo:
            combo_action = self._combo_engine.tick(now)
            if combo_action:
                self._execute_combo_action_str(combo_action)

        # 10. RL: Compute reward, store transition, encode new state
        # Accumulate attack on every tick for the RL accumulator
        if should_attack:
            self._rl_accum_attacked = True
        # Throttle RL to every 3rd tick - saves ~1ms per skipped tick
        if self._tick_count % 3 == 0:
            try:
                self._rl_step(bb, data, brawler, movement, should_attack)
            except Exception as e:
                if self._tick_count % 500 == 0:
                    print(f"[RL] _rl_step error (tick {self._tick_count}): {e}")

        # 10b. Hybrid RL partial override (movement only) with safety guardrails.
        # Keep BT authoritative in dangerous/critical states.
        if self._rl_mode == "hybrid":
            rl_movement = bb.get("rl.suggested_movement", "")
            allow_override, gate_reason = self._should_apply_rl_movement_override(bb, movement, rl_movement)
            if allow_override:
                movement = self._normalize_movement_keys(rl_movement)
                bb.set("decision.movement", movement)
                self._rl_move_override_applied += 1
                if gate_reason == "critical_evasive_allowed":
                    self._rl_move_override_critical_applied += 1
            elif rl_movement:
                self._rl_move_override_blocked += 1
                if gate_reason == "attack_window_block":
                    self._rl_move_override_blocked_attack_window += 1
                elif gate_reason == "pattern_pressure_block":
                    self._rl_move_override_blocked_pattern_pressure += 1

        # 10c. Final movement smoothing to prevent frame-to-frame direction flapping.
        movement = self._stabilize_movement(bb, movement)
        bb.set("decision.movement", movement)

        # 11. Debug logging
        if self._debug_print and self._tick_count % self._log_every_n_ticks == 0:
            self._debug_log(bb, movement, reason, should_attack)

        return movement

    def _should_apply_rl_movement_override(self, bb: Blackboard,
                                           bt_movement: str,
                                           rl_movement: str) -> tuple[bool, str]:
        """Decide if RL movement suggestion can safely override BT movement."""
        if not rl_movement or rl_movement == bt_movement:
            return False, "same_or_empty"

        if any(ch not in "WASD" for ch in rl_movement):
            return False, "invalid_keys"

        hp = float(bb.get("player.hp", 100) or 100)
        retreat_hp = float(bb.get("style.hp_retreat", 45) or 45)

        if hp <= (retreat_hp + 5):
            return False, "low_hp"
        if bool(bb.get("dodge.active", False)):
            return False, "dodge_active"
        if _storm_threat_active(bb):
            return False, "gas_or_storm"

        projectiles = bb.get("projectiles", []) or []
        urgent_threat = any(0 < int(p.get("frames_to_impact", -1)) < 6 for p in projectiles)
        if urgent_threat:
            return False, "urgent_projectile"

        attack_soon = float(bb.get("enemy.predicted_attack_soon", 0.0) or 0.0)
        pattern_pressure = float(bb.get("enemy.pattern_pressure", 0.0) or 0.0)
        team_aggr = float(bb.get("enemy.team_aggression", 0.5) or 0.5)

        attack_window_critical = attack_soon > 0.5
        pattern_critical = pattern_pressure > (0.60 - 0.12 * min(max(team_aggr, 0.0), 1.0))

        if attack_window_critical or pattern_critical:
            closest = bb.get("enemies_closest")
            player_pos = bb.get("player.pos", (0.0, 0.0)) or (0.0, 0.0)
            enemy_pos = closest.get("pos") if isinstance(closest, dict) else None
            if enemy_pos is None:
                if attack_window_critical:
                    return False, "attack_window_block"
                return False, "pattern_pressure_block"

            rel_x = float(enemy_pos[0]) - float(player_pos[0])
            rel_y = float(enemy_pos[1]) - float(player_pos[1])
            rel_len = math.hypot(rel_x, rel_y)
            if rel_len < 1e-6:
                if attack_window_critical:
                    return False, "attack_window_block"
                return False, "pattern_pressure_block"

            nx = rel_x / rel_len
            ny = rel_y / rel_len

            def _move_vec(keys: str) -> tuple[float, float]:
                dx = 0.0
                dy = 0.0
                for ch in str(keys or ""):
                    if ch == "W":
                        dy -= 1.0
                    elif ch == "S":
                        dy += 1.0
                    elif ch == "A":
                        dx -= 1.0
                    elif ch == "D":
                        dx += 1.0
                ln = math.hypot(dx, dy)
                if ln <= 1e-6:
                    return 0.0, 0.0
                return dx / ln, dy / ln

            rlx, rly = _move_vec(rl_movement)
            btx, bty = _move_vec(bt_movement)

            rl_toward = (rlx * nx + rly * ny)
            bt_toward = (btx * nx + bty * ny)
            rl_strafe = abs((-ny * rlx + nx * rly))

            # In critical windows RL override is only accepted if it is evasive
            # and no more forward-committing than BT.
            evasive_ok = (rl_toward <= 0.05) or (rl_strafe >= 0.60)
            less_committing_than_bt = rl_toward <= (bt_toward + 0.05)
            if evasive_ok and less_committing_than_bt:
                return True, "critical_evasive_allowed"
            if attack_window_critical:
                return False, "attack_window_block"
            return False, "pattern_pressure_block"

        return True, "allowed"

    def _debug_log(self, bb: Blackboard, movement: str, reason: str, attacked: bool):
        """Print debug info for live testing."""
        player_pos = bb.get("player.pos", (0, 0))
        player_hp = bb.get("player.hp", -1)
        enemies_count = bb.get("enemies_count", 0)
        closest = bb.get("enemies_closest")
        closest_dist = closest.get("distance", "?") if closest else "N/A"
        
        # Gas/storm info
        gas_active = bb.get("map.gas_active", False)
        in_storm = bb.get("map.in_storm", False)
        effective_storm = _storm_threat_active(bb)
        self._last_in_storm = effective_storm  # Track for forced death penalty
        storm_flag = " GAS!" if effective_storm and gas_active else (" STORM!" if effective_storm and in_storm else "")

        ammo = bb.get("player.ammo", -1)
        hittable = closest.get("hittable", "?") if closest else "N/A"
        atk_range = bb.get("brawler.attack_range_scaled", "?")
        print(f"[BT #{self._tick_count}] mv={movement} atk={attacked} "
              f"hp={player_hp} ammo={ammo} enemies={enemies_count} dist={closest_dist} hit={hittable} r={atk_range}{storm_flag}")
        if reason:
            print(f"         reason: {reason}")

    def _update_subsystems(self, data: dict, frame, brawler: str):
        """Update all optional subsystems with current frame data.
        
        PERFORMANCE: Heavy subsystems are throttled to reduce CPU load.
        """
        bb = self.blackboard
        player_pos = bb.get("player.pos", (0, 0))
        enemies = data.get('enemy', []) or []
        now = time.time()

        # Enemy tracker (every frame - critical for combat)
        if self._enemy_tracker:
            self._enemy_tracker.update(
                enemies, player_pos,
                bb.get("player.hp", 100), frame
            )
            # Enrich blackboard with tracked data
            tracked = self._enemy_tracker.get_tracked_enemies()
            bb.set("tracked_enemies", tracked)
            safest = self._enemy_tracker.get_safest_target(player_pos)
            if safest:
                bb.set("safest_target", safest)

        # Projectile detector (throttled to ~10Hz to save CPU - was 20Hz)
        if self._projectile_detector and frame is not None:
            import numpy as np
            if now - getattr(self, '_last_projectile_time', 0) < 0.1:  # 10Hz instead of 20Hz
                # Re-use cached projectiles
                projectiles = bb.get("projectiles", [])
            else:
                self._last_projectile_time = now
                arr = np.asarray(frame) if not isinstance(frame, np.ndarray) else frame
                projectiles = self._projectile_detector.detect(arr, player_pos)
            bb.set("projectiles", projectiles)
            # Find the most threatening projectile (must be close enough to matter)
            # Only dodge if projectile will hit within 8 frames (~0.25s at 30fps)
            threats = [p for p in projectiles 
                       if 0 < p.get("frames_to_impact", -1) < 8]
            threats.sort(key=lambda p: p["frames_to_impact"])
            if threats:
                worst = threats[0]
                dd = self._projectile_detector.get_dodge_direction(worst, player_pos)
                keys = self._projectile_detector.direction_to_keys(dd[0], dd[1])
                bb.set("dodge.active", True)
                bb.set("dodge.direction", keys)
            else:
                bb.set("dodge.active", False)

        # Spatial memory (throttled to 5Hz - map doesn't change fast)
        if self._spatial_memory:
            if now - getattr(self, '_last_spatial_time', 0) >= 0.2:
                self._last_spatial_time = now
                walls = data.get('wall', []) or []
                bushes = getattr(self.play, 'last_bush_data', []) or []
                self._spatial_memory.update_from_detections(walls, bushes)
                self._spatial_memory.update_visibility(player_pos)

        # Opponent model (throttled to 10Hz - behavior patterns don't change fast)
        if self._opponent_model:
            if now - getattr(self, '_last_opponent_time', 0) >= 0.1:
                self._last_opponent_time = now
                enemy_dicts = bb.get("enemies", [])
                player_damage_taken = float(bb.get("player.damage_taken", 0.0) or 0.0)
                self._opponent_model.update(
                    enemy_dicts,
                    player_pos,
                    time.time(),
                    player_damage_taken=player_damage_taken,
                )
                counter = self._opponent_model.get_counter_strategy()
                bb.set("counter_strategy", counter)
                bb.set("enemy.team_style", getattr(self._opponent_model, 'team_style', 'unknown'))
                bb.set("enemy.team_aggression", float(getattr(self._opponent_model, 'team_aggression', 0.5) or 0.5))
                profiles = list(getattr(self._opponent_model, "profiles", {}).values())
                if profiles:
                    attack_in = 999.0
                    safety_window = 0.0
                    for idx in range(len(profiles)):
                        pred_i = self._opponent_model.get_enemy_prediction(idx)
                        attack_i = float(pred_i.get("attack_in_seconds", 999.0) or 999.0)
                        safety_i = float(pred_i.get("safety_window", 0.0) or 0.0)
                        if attack_i < attack_in:
                            attack_in = attack_i
                            safety_window = safety_i
                else:
                    pred = self._opponent_model.get_enemy_prediction(0)
                    attack_in = float(pred.get("attack_in_seconds", 999.0) or 999.0)
                    safety_window = float(pred.get("safety_window", 0.0) or 0.0)
                bb.set("enemy.predicted_attack_in", attack_in)
                bb.set("enemy.safety_window", safety_window)
                bb.set("enemy.predicted_attack_soon", 1.0 if attack_in <= 0.9 else 0.0)

                try:
                    profiles = list(getattr(self._opponent_model, "profiles", {}).values())
                    approach = 0
                    retreat = 0
                    strafe = 0
                    total = 0
                    for profile in profiles:
                        approach += int(getattr(profile, "approach_count", 0) or 0)
                        retreat += int(getattr(profile, "retreat_count", 0) or 0)
                        strafe += int(getattr(profile, "strafe_count", 0) or 0)
                        total += (
                            int(getattr(profile, "approach_count", 0) or 0)
                            + int(getattr(profile, "retreat_count", 0) or 0)
                            + int(getattr(profile, "strafe_count", 0) or 0)
                            + int(getattr(profile, "idle_count", 0) or 0)
                        )
                    if total > 0:
                        strafe_ratio = float(strafe) / float(total)
                        approach_ratio = float(approach) / float(total)
                        pattern_pressure = min(1.0, (strafe_ratio * 1.3) + (approach_ratio * 0.7))
                    else:
                        strafe_ratio = 0.0
                        approach_ratio = 0.0
                        pattern_pressure = 0.0
                except Exception:
                    strafe_ratio = 0.0
                    approach_ratio = 0.0
                    pattern_pressure = 0.0

                bb.set("enemy.strafe_ratio", strafe_ratio)
                bb.set("enemy.approach_ratio", approach_ratio)
                bb.set("enemy.pattern_pressure", pattern_pressure)

        # Visual ammo reading (throttled to 15Hz - ammo changes slowly)
        if self._ammo_reader:
            if now - getattr(self, '_last_ammo_time', 0) >= 0.066:
                self._last_ammo_time = now
                visual_ammo = self._ammo_reader.read_ammo(frame)
                if visual_ammo >= 0:
                    try:
                        prev_ammo = int(bb.get("player.ammo", visual_ammo) or visual_ammo)
                    except Exception:
                        prev_ammo = 0
                    try:
                        max_ammo = int(bb.get("player.max_ammo", 3) or 3)
                    except Exception:
                        max_ammo = 3
                    try:
                        visual_i = int(visual_ammo)
                    except Exception:
                        visual_i = prev_ammo
                    clamped_visual = max(0, min(max_ammo, visual_i))
                    # Guard against abrupt false drops from UI/OCR noise.
                    # A real ammo decrease larger than 1 pip in one 66ms read is unlikely.
                    if clamped_visual < prev_ammo - 1:
                        clamped_visual = prev_ammo - 1
                    clamped_visual = max(0, min(max_ammo, clamped_visual))
                    bb.set("player.ammo", clamped_visual)

        # Combo engine: check if should start a combo
        if self._combo_engine and not self._combo_engine._active_combo:
            playstyle = bb.get("brawler.playstyle", "fighter")
            self._combo_engine.try_start(bb, playstyle)

    def _execute_attack(self, bb: Blackboard, attack_type: str):
        """Execute the attack decision with smart aim selection.
        
        Automatically chooses manual aim (lead shots) vs auto-aim based on:
        - Enemy distance (far = manual, close = auto)
        - Enemy speed (fast = manual for lead)
        - Brawler playstyle (snipers/throwers benefit most)
        - ManualAimer availability
        - Bush target position (aimed shot at last known pos)
        
        For hold-attack brawlers (Hank, Angelo): holds the button to charge,
        then releases after the configured charge duration.
        """
        p = self.play
        if p is None:
            return  # No play instance - test / headless mode

        # === HOLD-ATTACK (CHARGE) BRAWLER LOGIC ===
        # Check once per brawler selection whether this brawler charges attacks
        current_brawler = getattr(p, "current_brawler", None)
        brawlers_info = getattr(p, "brawlers_info", {})
        if not self._is_hold_brawler and current_brawler:
            self._is_hold_brawler = p.must_brawler_hold_attack(
                current_brawler, brawlers_info)
            if self._is_hold_brawler:
                self._hold_attack_duration = brawlers_info.get(
                    current_brawler, {}).get('hold_attack', 3)
                print(f"[BT] Hold-attack brawler detected: {current_brawler} "
                      f"(charge={self._hold_attack_duration}s)")

        if self._is_hold_brawler:
            now = time.time()
            hold_total = self._hold_attack_duration + getattr(
                p, 'seconds_to_hold_attack_after_reaching_max', 1.5)

            if self._hold_attack_start is None:
                # --- Start charging: press down, do NOT release ---
                try:
                    p.attack(touch_up=False, touch_down=True)
                except Exception:
                    self._hold_attack_start = None
                    return
                self._hold_attack_start = now
                if self._debug_print:
                    print(f"[BT] HOLD-ATTACK START: charging {hold_total:.1f}s")
                return

            elapsed = now - self._hold_attack_start
            if elapsed >= hold_total:
                # --- Fully charged: release ---
                try:
                    p.attack(touch_up=True, touch_down=False)
                except Exception:
                    pass
                try:
                    p._spend_ammo()
                except Exception:
                    pass
                self._hold_attack_start = None
                if self._debug_print:
                    print(f"[BT] HOLD-ATTACK RELEASE after {elapsed:.1f}s")
                return

            # --- Still charging: do nothing (keep held) ---
            if self._debug_print and self._tick_count % self._log_every_n_ticks == 0:
                print(f"[BT] HOLD-ATTACK CHARGING: {elapsed:.1f}/{hold_total:.1f}s")
            return
        # === END HOLD-ATTACK ===

        closest = bb.get("enemies_closest")
        player_pos = bb.get("player.pos", (0, 0))
        playstyle = bb.get("brawler.playstyle", "fighter")
        attack_range_scaled = bb.get("brawler.attack_range_scaled", 300)
        
        # bUSH CHECK: Aim at bush position if set
        bush_target = bb.get("decision.bush_target_pos")
        aimer = self._manual_aimer
        if bush_target and aimer is not None:
            try:
                aimer.aimed_attack(
                    player_pos, bush_target, lead_offset=(0, 0),
                    playstyle=playstyle, attack_range=attack_range_scaled
                )
                p._spend_ammo()
                dist = int(math.hypot(bush_target[0] - player_pos[0],
                                      (bush_target[1] - player_pos[1]) * 1.25))
                print(f"[BT] BUSH SHOT: aimed at ({int(bush_target[0])},{int(bush_target[1])}) dist={dist}px")
                bb.set("decision.bush_target_pos", None)  # Clear after shooting
                return
            except Exception as e:
                if self._debug_print:
                    print(f"[BT] BUSH SHOT FAILED: {e}")
                # Fall through to auto-aim
        
        # Decide: manual aim or auto-aim?
        use_manual = False
        distance = 0
        enemy_speed = 0.0
        if aimer is not None and closest:
            distance = closest.get("distance", 0)
            enemy_speed = bb.get("enemy.speed", 0.0)
            use_manual = aimer.should_use_manual_aim(
                distance, enemy_speed, playstyle
            )
            if playstyle == "thrower" and distance >= 60:
                use_manual = True
        
        if use_manual and closest and aimer is not None:
            # mANUAL AIM: Lead the target
            try:
                lead = p._get_lead_offset(distance, playstyle)

                # Enhance lead with EnemyTracker data if available
                # The tracker has per-entity velocity with 30-position history
                # (more stable than play.py's 8-position global tracking)
                tracked = bb.get("tracked_enemies", [])
                if tracked and distance > 150:
                    # Find matching tracked enemy (nearest to closest["pos"])
                    cpos = closest["pos"]
                    best_track = None
                    best_tdist = 100  # max 100px association
                    for te in tracked:
                        td = math.hypot(te["pos"][0] - cpos[0], te["pos"][1] - cpos[1])
                        if td < best_tdist:
                            best_tdist = td
                            best_track = te
                    if best_track is not None:
                        # Cross-validate: blend tracker velocity with play.py velocity
                        tvx, tvy = best_track["velocity"]
                        tracker_speed = best_track["speed"]
                        if tracker_speed > 40:
                            pvx, pvy = p._enemy_velocity_smooth
                            # 40% tracker + 60% play.py (play.py has isometric correction)
                            blend_vx = 0.4 * tvx + 0.6 * pvx
                            blend_vy = 0.4 * tvy + 0.6 * pvy
                            # Recompute lead with blended velocity
                            brawler_info = p.brawlers_info.get(p.current_brawler, {})
                            proj_speed = brawler_info.get('projectile_speed', 800)
                            if proj_speed > 0 and distance > 0:
                                tt = distance / proj_speed
                                blended_lead_x = blend_vx * tt
                                blended_lead_y = blend_vy * tt
                                # Use stronger lead (max magnitude) to avoid under-prediction
                                lead_mag = math.hypot(lead[0], lead[1])
                                blend_mag = math.hypot(blended_lead_x, blended_lead_y)
                                if blend_mag > lead_mag:
                                    # Tracker predicts more movement - use 70/30 blend toward tracker
                                    lead = (
                                        0.3 * lead[0] + 0.7 * blended_lead_x,
                                        0.3 * lead[1] + 0.7 * blended_lead_y
                                    )
                                # else: keep original play.py lead (it's already stronger)

                aimer.aimed_attack(
                    player_pos, closest["pos"], lead_offset=lead,
                    playstyle=playstyle, attack_range=attack_range_scaled
                )
                p._spend_ammo()
                if self._debug_print and self._tick_count % self._log_every_n_ticks == 0:
                    print(f"[BT] AIMED SHOT: dist={int(distance)}px speed={int(enemy_speed):.0f}px/s "
                          f"lead=({int(lead[0])},{int(lead[1])}) playstyle={playstyle}")
                return
            except Exception as e:
                # ManualAimer failed - fall through to auto-aim
                if self._debug_print:
                    print(f"[BT] AIMED SHOT FAILED, fallback auto-aim: {e}")
        
        # aUTO-AIM: Tap attack button (always works)
        try:
            p.attack()
            p._spend_ammo()
        except Exception:
            pass
        if self._debug_print and self._tick_count % self._log_every_n_ticks == 0:
            dist = int(closest.get('distance', 0)) if closest else 0
            print(f"[BT] AUTO-AIM: dist={dist}px")

    def _execute_super(self, bb: Blackboard):
        """Execute super with aimed shot when beneficial."""
        p = self.play
        if p is None:
            return
        
        closest = bb.get("enemies_closest")
        player_pos = bb.get("player.pos", (0, 0))
        playstyle = bb.get("brawler.playstyle", "fighter")
        super_range_scaled = bb.get("brawler.super_range_scaled", bb.get("brawler.attack_range_scaled", 300))
        
        # For ranged brawlers, aim the super
        aimer = self._manual_aimer
        if aimer is not None and closest:
            distance = closest.get("distance", 0)
            enemy_speed = bb.get("enemy.speed", 0.0)
            # Aim super if enemy is far enough or moving fast
            if distance > 150 or enemy_speed > 80:
                lead = p._get_lead_offset(distance, playstyle)
                aimer.aimed_super(
                    player_pos, closest["pos"], lead_offset=lead,
                    playstyle=playstyle, attack_range=super_range_scaled
                )
                print(f"[BT] AIMED SUPER: dist={int(distance)}px lead=({int(lead[0])},{int(lead[1])})")
                return
        
        # Fallback: auto-aim super (tap)
        if p.use_super():
            print(f"[BT] SUPER USED (auto-aim)")

    def _execute_combo_action_str(self, action_name: str):
        """Execute a combo action by name (returned by ComboEngine.tick)."""
        p = self.play
        if p is None:
            return

        if action_name == "attack":
            p.attack()
            p._spend_ammo()
        elif action_name == "super":
            p.use_super()
        elif action_name == "gadget":
            p.window_controller.press_key("G")
        elif action_name == "wait":
            pass  # Do nothing this tick

    def _rl_step(self, bb: Blackboard, data: dict, brawler: str,
                  movement: str, attacked: bool):
        """Handle RL state encoding, reward computation, and transition storage.

        In BT mode: observes BT actions and trains the policy to imitate them
        via proper log_prob/value evaluation (behavioral cloning + reward shaping).
        In hybrid mode: RL suggests actions, BT is safety net.
        """
        if not self._state_encoder or not self._rl_trainer:
            return

        # Encode current state
        try:
            current_state = self._state_encoder.encode(bb)
        except Exception as e:
            if self._tick_count % 500 == 0:
                print(f"[RL] Encode error: {e}")
            return

        # Compute step reward using accumulated events since last RL tick
        # (RL runs every 3rd tick, so we collect damage/kills across all 3 frames)
        reward = 0.0
        if self._reward_calculator:
            try:
                # Get closest enemy info for proximity rewards
                closest = bb.get("enemies_closest")
                closest_dist = closest.get("distance", 9999) if closest else 9999
                closest_hp = closest.get("hp_pct", 100) if closest else 100
                
                reward = self._reward_calculator.calculate(
                    hp_diff=bb.get("player.hp", 100) - bb.get("player.prev_hp", 100),
                    enemy_hp_diff=0,  # Handled via accumulated damage_dealt
                    attacked=attacked or self._rl_accum_attacked,
                    killed=self._rl_accum_kills > 0,
                    died=self._rl_accum_deaths > 0,
                    damage_dealt=self._rl_accum_damage_dealt,
                    damage_taken=self._rl_accum_damage_taken,
                    in_storm=_storm_threat_active(bb),
                    player_hp_pct=bb.get("player.hp", 100),
                    retreat_hp_threshold=bb.get("style.hp_retreat", 45),
                    attack_range=bb.get("brawler.attack_range_scaled", 400),
                    player_pos=bb.get("player.pos", (0, 0)),
                    closest_enemy_dist=closest_dist,
                    closest_enemy_hp_pct=closest_hp,
                )
            except Exception:
                reward = 0.0
        # Reset accumulators after consumption
        self._rl_accum_damage_dealt = 0.0
        self._rl_accum_damage_taken = 0.0
        self._rl_accum_kills = 0
        self._rl_accum_deaths = 0
        self._rl_accum_attacked = False

        # Store transition from previous state -> current state
        if self._last_state is not None:
            done = bb.get("match.ended", False) or bb.get("player.just_died", False)
            self._rl_trainer.store_transition(
                state=self._last_state,
                action=self._last_action,
                reward=reward,
                done=done,
                log_prob=self._last_log_prob,
                value=self._last_value,
            )

        # Determine action + evaluate with model for proper log_prob & value
        from rl.trainer import encode_action, decode_action

        if self._rl_mode == "hybrid" and self._rl_trainer.is_available:
            # Hybrid: RL selects actions, BT provides safety net
            # Multi-head: returns (move,atk,abi), (lp,lp,lp), value
            actions, log_probs, value = self._rl_trainer.select_action(current_state)
            self._last_action = actions         # tuple (m, a, ab)
            self._last_log_prob = log_probs     # tuple (lp, lp, lp)
            self._last_value = value
            rl_move, rl_attack, rl_ability = decode_action(actions)
            bb.set("rl.suggested_movement", rl_move)
            bb.set("rl.suggested_attack", rl_attack)
            bb.set("rl.suggested_ability", rl_ability)
        else:
            # BT mode: encode the BT-chosen action, then evaluate it with the
            # model to get proper log_prob and value (needed for PPO training).
            attack_str = "auto_aim" if attacked else "none"
            ability_str = "none"
            if bb.get("used.super"):
                ability_str = "super"
            elif bb.get("used.gadget"):
                ability_str = "gadget"
            action_tuple = encode_action(movement, attack_str, ability_str)
            self._last_action = action_tuple  # (move_idx, attack_idx, ability_idx)

            # Evaluate BT's action through the RL model for proper log_prob + value
            if self._rl_trainer.is_available:
                try:
                    import torch
                    from torch.distributions import Categorical
                    with torch.no_grad():
                        st = torch.FloatTensor(current_state).unsqueeze(0)
                        move_p, attack_p, ability_p, value = self._rl_trainer.model(st)
                        m_idx, a_idx, ab_idx = action_tuple
                        move_lp = Categorical(move_p).log_prob(torch.tensor([m_idx])).item()
                        attack_lp = Categorical(attack_p).log_prob(torch.tensor([a_idx])).item()
                        ability_lp = Categorical(ability_p).log_prob(torch.tensor([ab_idx])).item()
                        self._last_log_prob = (move_lp, attack_lp, ability_lp)
                        self._last_value = value.item()
                except Exception:
                    self._last_log_prob = (0.0, 0.0, 0.0)
                    self._last_value = 0.0
            else:
                self._last_log_prob = (0.0, 0.0, 0.0)
                self._last_value = 0.0

        self._last_state = current_state

        # Debug: log buffer growth periodically
        if self._tick_count % 300 == 0:
            buf_sz = len(self._rl_trainer.buffer)
            components = ""
            if self._reward_calculator and self._reward_calculator._reward_components:
                components = " " + str(self._reward_calculator._reward_components)
            total_ep = self._reward_calculator._total_episode_reward if self._reward_calculator else 0
            print(f"[RL] tick={self._tick_count} buf={buf_sz} r={reward:.3f} ep={total_ep:.1f}{components}")

    def get_debug_info(self) -> dict:
        """Return debug info for visual overlay."""
        info = {
            "tree_path": self._last_tree_path,
            "tick_count": self._tick_count,
            "reason": self.blackboard.get("decision.reason", ""),
            "enemy_tracker": (self._enemy_tracker.get_summary()
                              if self._enemy_tracker else "N/A"),
            "opponent_model": (self._opponent_model.get_summary()
                               if self._opponent_model else "N/A"),
            "spatial_memory": self._spatial_memory is not None,
            "combo_active": (self._combo_engine._active_combo is not None
                             if self._combo_engine else False),
            "aim_stats": (self._manual_aimer.get_stats()
                          if self._manual_aimer else {}),
            # Reward/stats info
            "match_kills": self._match_kills,
            "match_deaths": self._match_deaths,
            "session_kills": self._session_kills,
            "session_deaths": self._session_deaths,
            "total_damage_dealt": int(self._session_damage_dealt),
            "total_damage_taken": int(self._session_damage_taken),
            "reward_score": self._get_reward_score(),
            "aggression_modifier": self._get_adaptive_aggression(),
        }
        # RL stats
        if self._rl_trainer and self._rl_trainer.is_available:
            info["rl_stats"] = self._rl_trainer.get_stats_string()
            info["rl_episodes"] = self._rl_trainer.total_episodes
            info["rl_buffer_size"] = len(self._rl_trainer.buffer)
        # Reward calculator stats
        if self._reward_calculator:
            summary = self._reward_calculator.get_episode_summary()
            info["reward_total"] = summary.get("total_reward", 0.0)
            info["reward_components"] = summary.get("last_components", {})
        return info
    
    def _get_reward_score(self) -> float:
        """Calculate a running reward score for the current match.
        
        Heavily weighted toward kills (main objective) and against deaths.
        + big points for kills, kill streaks
        - big points for deaths
        + small points for damage dealt
        """
        score = 0.0
        score += self._match_kills * 15.0       # Kill is the main goal
        score -= self._match_deaths * 12.0      # Death is very bad
        score += self._session_damage_dealt / 1000 * 0.8
        score -= self._session_damage_taken / 1000 * 0.6
        # Bonus for being alive with kills
        if self._match_deaths == 0 and self._match_kills > 0:
            score += 8.0
        return round(score, 1)
    
    def _get_adaptive_aggression(self) -> float:
        """Calculate adaptive aggression modifier based on match performance.
        
        Range: 0.5 (very cautious) to 1.5 (very aggressive)
        - Kills increase aggression (bot learns to be aggressive)
        - Deaths decrease aggression (bot learns to be careful)
        - High damage dealt rewards continued aggression
        """
        base = 1.0
        
        # Kill bonus: +0.1 per kill (max +0.3)
        kill_bonus = min(self._match_kills * 0.1, 0.3)
        
        # Death penalty: -0.08 per death (max -0.24) - reduced to prevent death spiral
        death_penalty = min(self._match_deaths * 0.08, 0.24)
        
        # Damage ratio bonus: if we dealt more than we took
        if self._session_damage_taken > 0:
            ratio = self._session_damage_dealt / max(1, self._session_damage_taken)
            if ratio > 1.5:
                base += 0.1  # We're winning trades
            elif ratio < 0.5:
                base -= 0.1  # We're losing trades
        
        # No kills penalty: if we haven't killed anyone after many ticks
        if self._tick_count > 200 and self._match_kills == 0:
            base -= 0.1  # Need to be more careful/strategic
        
        aggression = base + kill_bonus - death_penalty
        return max(0.7, min(1.5, round(aggression, 2)))  # Floor 0.7 (was 0.5) - stay aggressive

    def _compute_rl_kpi_reward_adjustment(self, summary: dict) -> float:
        """Small episode-level shaping based on RL override safety KPIs.

        Positive: RL critical-window overrides that were allowed as evasive.
        Negative: repeated blocks in predicted-attack windows (risk-seeking RL proposals).
        """
        if not isinstance(summary, dict):
            return 0.0

        critical_applied = int(summary.get("rl_move_override_critical_applied", 0) or 0)
        block_attack_window = int(summary.get("rl_move_override_blocked_attack_window", 0) or 0)
        block_pattern_pressure = int(summary.get("rl_move_override_blocked_pattern_pressure", 0) or 0)
        attack_soon_frames = int(summary.get("enemy_attack_soon_frames", 0) or 0)
        pattern_pressure_frames = int(summary.get("enemy_pattern_pressure_frames", 0) or 0)

        high_threat_scale = min(1.0, (attack_soon_frames + 0.7 * pattern_pressure_frames) / 220.0)
        bonus = min(critical_applied, 30) * (
            self._kpi_adj_bonus_base + self._kpi_adj_bonus_threat_scale * high_threat_scale
        )

        effective_attack_blocks = max(0, block_attack_window - max(1, critical_applied // 2))
        attack_block_penalty = min(effective_attack_blocks, 45) * (
            self._kpi_adj_attack_block_base_penalty
            + self._kpi_adj_attack_block_threat_penalty * high_threat_scale
        )
        pattern_block_penalty = min(block_pattern_pressure, 50) * self._kpi_adj_pattern_block_penalty

        adjustment = bonus - attack_block_penalty - pattern_block_penalty
        clip_abs = max(0.1, float(self._kpi_adj_clip_abs))
        return float(max(-clip_abs, min(clip_abs, adjustment)))

    def reset_match(self, match_won: bool = False):
        """Reset all subsystems for a new match.

        """
        # In Showdown, LOSS = eliminated = death.  If the final death wasn't
        # detected by the respawn sensor, force-count it so RL learns properly.
        if not match_won and self._match_deaths == 0:
            self._match_deaths = 1
            # Inject the per-frame death penalty into the RL reward calculator
            if self._reward_calculator:
                in_gas = getattr(self, '_last_in_storm', False)
                self._reward_calculator.calculate(
                    died=True,
                    in_storm=in_gas,
                )
                tag = " (GAS)" if in_gas else ""
                print(f"[REWARD] Forced final death penalty{tag} (elimination)")

        # Log match result with reward info
        reward_score = self._get_reward_score()
        kd = f"{self._match_kills}/{self._match_deaths}"
        dmg = f"dealt={int(self._session_damage_dealt)} taken={int(self._session_damage_taken)}"
        result = "WIN" if match_won else "LOSS"
        print(f"[REWARD] Match {result} | K/D: {kd} | {dmg} | Score: {reward_score}")
        
        # Add match-level reward
        # Pass kill count so reward calculator can distinguish
        # active wins (1+ kills) from passive wins (0 kills, carried).
        if self._reward_calculator:
            # Sync kill count from play.py into reward calculator so
            # match reward scales properly with contribution.
            if self.play:
                play_kills = getattr(self.play, '_enemies_killed_this_match', 0)
                if play_kills > self._reward_calculator._kills:
                    self._reward_calculator._kills = play_kills
            self._reward_calculator.add_match_result(
                won=match_won,
                trophy_delta=8 if match_won else -4
            )

        # RL: End episode + train
        if self._reward_calculator and self._rl_trainer:
            try:
                buf_size = len(self._rl_trainer.buffer)
                print(f"[RL] End-of-match: buffer={buf_size} transitions, ticks={self._tick_count}")

                summary = self._reward_calculator.episode_summary(won=match_won)
                summary["rl_move_override_applied"] = int(self._rl_move_override_applied)
                summary["rl_move_override_blocked"] = int(self._rl_move_override_blocked)
                summary["rl_move_override_blocked_attack_window"] = int(self._rl_move_override_blocked_attack_window)
                summary["rl_move_override_blocked_pattern_pressure"] = int(self._rl_move_override_blocked_pattern_pressure)
                summary["rl_move_override_critical_applied"] = int(self._rl_move_override_critical_applied)
                summary["peek_active_frames"] = int(self._peek_active_frames)
                summary["water_pressure_frames"] = int(self._water_pressure_frames)
                summary["enemy_pattern_pressure_frames"] = int(self._enemy_pattern_pressure_frames)
                summary["enemy_attack_soon_frames"] = int(self._enemy_attack_soon_frames)
                summary["kpi_profile"] = str(getattr(self, "_kpi_adj_profile", "balanced") or "balanced")
                summary["kpi_adjuster"] = {
                    "bonus_base": float(self._kpi_adj_bonus_base),
                    "bonus_threat_scale": float(self._kpi_adj_bonus_threat_scale),
                    "attack_block_base_penalty": float(self._kpi_adj_attack_block_base_penalty),
                    "attack_block_threat_penalty": float(self._kpi_adj_attack_block_threat_penalty),
                    "pattern_block_penalty": float(self._kpi_adj_pattern_block_penalty),
                    "clip_abs": float(self._kpi_adj_clip_abs),
                }
                kpi_reward_adjustment = self._compute_rl_kpi_reward_adjustment(summary)
                if abs(kpi_reward_adjustment) > 1e-6:
                    summary["total_reward"] = float(summary.get("total_reward", 0.0) or 0.0) + kpi_reward_adjustment
                    summary["total_raw_reward"] = float(summary.get("total_raw_reward", 0.0) or 0.0) + kpi_reward_adjustment
                summary["kpi_reward_adjustment"] = float(kpi_reward_adjustment)
                tobs_match_damage = 0.0
                tobs_match_delta = 0
                tobs_total_damage = self._last_tobs_total_damage
                tobs_total_matches = self._last_tobs_total_matches
                if self.play is not None:
                    try:
                        stats_info = getattr(self.play, '_stats_info', {}) or {}
                        tobs = stats_info.get('trophy_observer') if isinstance(stats_info, dict) else None
                        if tobs is not None:
                            sess = getattr(tobs, 'session_stats', {}) or {}
                            tobs_total_damage = float(sess.get('total_damage', 0) or 0)
                            tobs_total_matches = int(sess.get('total_matches', 0) or 0)
                            tobs_match_damage = max(0.0, tobs_total_damage - float(self._last_tobs_total_damage or 0.0))
                            tobs_match_delta = max(0, tobs_total_matches - int(self._last_tobs_total_matches or 0))
                    except Exception:
                        pass
                # Keep training stats consistent with BT session tracking.
                # RewardCalculator can undercount when detection is sparse,
                # but BT session totals are accumulated every tick.
                summary["damage_dealt"] = max(
                    float(summary.get("damage_dealt", 0.0) or 0.0),
                    float(self._session_damage_dealt or 0.0),
                )
                # End-screen OCR fallback (only if a new end-screen stat was recorded).
                if tobs_match_delta > 0:
                    summary["damage_dealt"] = max(
                        float(summary.get("damage_dealt", 0.0) or 0.0),
                        float(tobs_match_damage or 0.0),
                    )
                summary["damage_taken"] = max(
                    float(summary.get("damage_taken", 0.0) or 0.0),
                    float(self._session_damage_taken or 0.0),
                )
                total_reward = summary.get("total_reward", 0.0)

                # Pass rich data to trainer for comprehensive stats
                brawler_name = (self.blackboard.get("brawler.name", "") if self.blackboard else "") or ""
                # Fall back to last known brawler if BT didn't tick this match
                if not brawler_name:
                    brawler_name = getattr(self, '_last_known_brawler', 'unknown')
                else:
                    self._last_known_brawler = brawler_name

                # --- Skip aborted / ghost matches ---
                # If the BT barely ticked, no damage was exchanged, and
                # brawler is unknown, this was NOT a real match.  Recording
                # it would inject noise (-56 reward) into the RL model.
                dmg_dealt = summary.get("damage_dealt", 0.0)
                dmg_taken = summary.get("damage_taken", 0.0)
                is_ghost = (self._tick_count < 15
                            and dmg_dealt == 0
                            and dmg_taken == 0)
                if is_ghost:
                    print(f"[RL] SKIPPED ghost episode (ticks={self._tick_count}, "
                          f"brawler={brawler_name}, dmg=0/0) — not recording")
                    # Still clear the replay buffer so stale transitions
                    # don't leak into the next real episode.
                    self._rl_trainer.buffer.clear()
                else:
                    self._rl_trainer.end_episode(
                        total_reward,
                        summary=summary,
                        brawler=brawler_name,
                        match_won=match_won,
                    )
                    print(f"[RL] Episode reward: {total_reward:.2f} | "
                          f"K={summary.get('kills', 0)} D={summary.get('deaths', 0)} "
                          f"Acc={summary.get('accuracy', 0):.0%}")

                    # Train on collected experience
                    train_metrics = self._rl_trainer.update()
                    if train_metrics:
                        print(f"[RL] TRAINED! loss_p={train_metrics.get('policy_loss', 0):.4f} "
                              f"loss_v={train_metrics.get('value_loss', 0):.4f} "
                              f"entropy={train_metrics.get('entropy', 0):.2f} "
                              f"buf={train_metrics.get('buffer_size', 0)} "
                              f"updates={train_metrics.get('updates', 0)}")
                    else:
                        print(f"[RL] No update (buffer too small: {buf_size} < {self._rl_trainer.batch_size})")

                    # Save model after every match (for monitoring)
                    self._rl_trainer.save()
                    print(f"[RL] Model saved (episode {self._rl_trainer.total_episodes}, "
                          f"WR={self._rl_trainer.total_wins}/{self._rl_trainer.total_episodes})")

                # Update TrophyObserver baselines so next match can compute deltas correctly
                self._last_tobs_total_damage = float(tobs_total_damage or 0.0)
                self._last_tobs_total_matches = int(tobs_total_matches or 0)
            except Exception as e:
                print(f"[RL] Training error: {e}")

        # Reset RL state
        self._last_state = None
        if self._reward_calculator:
            self._reward_calculator.reset()

        # Reset match-level tracking
        self._session_damage_dealt = 0.0
        self._session_damage_taken = 0.0
        self._match_kills = 0
        self._match_deaths = 0
        self._last_play_kill_count = getattr(self.play, '_enemies_killed_this_match', 0) if self.play else 0
        self._prev_hp = 100.0
        self._prev_enemy_count = 0
        self._prev_enemy_hp = 100.0
        # Reset RL accumulators
        self._rl_accum_damage_dealt = 0.0
        self._rl_accum_damage_taken = 0.0
        self._rl_accum_kills = 0
        self._rl_accum_deaths = 0
        self._rl_accum_attacked = False
        self._rl_move_override_applied = 0
        self._rl_move_override_blocked = 0
        self._rl_move_override_blocked_attack_window = 0
        self._rl_move_override_blocked_pattern_pressure = 0
        self._rl_move_override_critical_applied = 0
        self._peek_active_frames = 0
        self._water_pressure_frames = 0
        self._enemy_pattern_pressure_frames = 0
        self._enemy_attack_soon_frames = 0

        # Reset core BT state
        self.blackboard = Blackboard()
        self._tick_count = 0
        self._hold_attack_start = None
        self._is_hold_brawler = False
        if self._enemy_tracker:
            self._enemy_tracker.reset()
        if self._projectile_detector:
            self._projectile_detector.reset()
        if self._spatial_memory:
            self._spatial_memory.reset()
        if self._combo_engine:
            self._combo_engine.reset()
        if self._opponent_model:
            self._opponent_model.reset()
        if self._manual_aimer:
            self._manual_aimer.reset()
        if self._ammo_reader:
            self._ammo_reader.reset()
        if self._hp_estimator:
            self._hp_estimator.reset()
