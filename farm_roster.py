from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from brawlstars_api import normalize_brawler_name, parse_player_profile_payload


FARM_STATE_PATH = Path("cfg") / "farm_roster.json"
FARM_MODES = {"manual", "api"}
FARM_STRATEGIES = {
    "lowest_first",
    "highest_first",
    "highest_trophies_first",
    "highest_winrate",
    "alphabetical",
    "manual_priority",
    "sequential",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def mask_api_key(api_key: str) -> str:
    value = str(api_key or "").strip()
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def normalize_farm_mode(value: str) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in FARM_MODES else "manual"


def normalize_farm_strategy(value: str) -> str:
    normalized = str(value or "").strip().lower()
    aliases = {
        "in_order": "alphabetical",
        "sequential": "alphabetical",
        "highest_trophies": "highest_first",
        "highest_trophies_first": "highest_first",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in FARM_STRATEGIES else "lowest_first"


def load_farm_state(path: Path | str = FARM_STATE_PATH) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    try:
        data = json.loads(file_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def save_farm_state(state: dict, path: Path | str = FARM_STATE_PATH) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(state if isinstance(state, dict) else {}, indent=4), encoding="utf-8")


def parse_api_roster(payload: dict, known_brawlers=None, player_tag: str = "") -> dict:
    parsed = parse_player_profile_payload(payload, player_tag)
    known_lookup = {
        normalize_brawler_name(name): str(name).strip().lower()
        for name in known_brawlers or []
        if str(name or "").strip()
    }
    roster = {}
    for normalized, entry in parsed["brawlers"].items():
        canonical = known_lookup.get(normalized, normalized)
        row = dict(entry)
        row["brawler"] = canonical
        row["displayName"] = str(entry.get("name") or canonical).title()
        row["owned"] = True
        row["included"] = True
        row["source"] = "api"
        roster[canonical] = row
    return {
        "player_name": parsed.get("player_name", ""),
        "player_tag": parsed.get("player_tag", ""),
        "brawlers": roster,
    }


def _safe_int(value, default=0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _history_winrate(history_entry) -> int:
    if not isinstance(history_entry, dict):
        return 50
    wins = _safe_int(history_entry.get("victory", 0), 0)
    defeats = _safe_int(history_entry.get("defeat", 0), 0)
    draws = _safe_int(history_entry.get("draw", 0), 0)
    total = wins + defeats + draws
    return round((wins / total) * 100) if total else 50


def _canonical_map(all_brawlers):
    return {
        normalize_brawler_name(name): str(name).strip().lower()
        for name in all_brawlers or []
        if str(name or "").strip()
    }


def _manual_entry_for(brawler, manual_roster, selected_lookup, scan_lookup):
    entry = dict(manual_roster.get(brawler, {}) if isinstance(manual_roster, dict) else {})
    selected = selected_lookup.get(brawler, {})
    scan = scan_lookup.get(brawler, {})
    trophies = entry.get("trophies", selected.get("trophies", scan.get("trophies", 0)))
    owned = entry.get("owned", entry.get("unlocked", True))
    included = entry.get("included", True)
    return {
        "brawler": brawler,
        "displayName": str(entry.get("displayName") or brawler).title(),
        "owned": bool(owned),
        "included": bool(included),
        "trophies": max(0, _safe_int(trophies, 0)),
        "priority": _safe_int(entry.get("priority", 0), 0),
        "source": str(entry.get("source", "manual")),
    }


def build_farm_plan(
    *,
    all_brawlers,
    selected_roster=None,
    scan_data=None,
    farm_state=None,
    target=500,
    strategy="lowest_first",
    excluded=None,
    mode="manual",
    history=None,
) -> dict:
    mode = normalize_farm_mode(mode)
    strategy = normalize_farm_strategy(strategy)
    target = max(0, _safe_int(target, 500))
    farm_state = farm_state if isinstance(farm_state, dict) else {}
    excluded = {
        normalize_brawler_name(name)
        for name in (excluded or [])
        if str(name or "").strip()
    }
    known_lookup = _canonical_map(all_brawlers)
    selected_lookup = {
        normalize_brawler_name(row.get("brawler", "")): row
        for row in selected_roster or []
        if isinstance(row, dict) and str(row.get("brawler", "")).strip()
    }
    scan_lookup = {
        normalize_brawler_name(name): entry
        for name, entry in (scan_data or {}).items()
        if isinstance(entry, dict)
    }
    history = history if isinstance(history, dict) else {}

    rows = []
    if mode == "api":
        api_roster = farm_state.get("api_roster", {})
        if isinstance(api_roster, dict):
            for key, value in api_roster.items():
                if not isinstance(value, dict):
                    continue
                canonical = known_lookup.get(normalize_brawler_name(key), normalize_brawler_name(key))
                rows.append({
                    "brawler": canonical,
                    "displayName": str(value.get("displayName") or value.get("name") or canonical).title(),
                    "owned": True,
                    "included": bool(value.get("included", True)),
                    "trophies": max(0, _safe_int(value.get("trophies", 0), 0)),
                    "highestTrophies": max(0, _safe_int(value.get("highestTrophies", value.get("trophies", 0)), 0)),
                    "power": _safe_int(value.get("power", 0), 0),
                    "rank": _safe_int(value.get("rank", 0), 0),
                    "priority": _safe_int(value.get("priority", 0), 0),
                    "source": "api",
                })
    else:
        manual_roster = farm_state.get("manual_roster", {})
        source_names = set(known_lookup.values())
        source_names.update(known_lookup.get(name, name) for name in selected_lookup)
        source_names.update(known_lookup.get(name, name) for name in scan_lookup)
        for brawler in sorted(name for name in source_names if name):
            rows.append(_manual_entry_for(brawler, manual_roster, selected_lookup, scan_lookup))

    for row in rows:
        key = normalize_brawler_name(row["brawler"])
        row["excluded"] = key in excluded
        row["winrate"] = _history_winrate(history.get(row["brawler"]) or history.get(key))
        row["target"] = target
        row["qualifies"] = (
            bool(row.get("owned", False))
            and bool(row.get("included", True))
            and not row["excluded"]
            and _safe_int(row.get("trophies", 0), 0) < target
        )
        if not row.get("owned", False):
            row["reason"] = "Not marked owned/unlocked"
        elif not row.get("included", True):
            row["reason"] = "Not included in this farm mode"
        elif row["excluded"]:
            row["reason"] = "Excluded from trophy farm"
        elif _safe_int(row.get("trophies", 0), 0) >= target:
            row["reason"] = "At or above target"
        else:
            row["reason"] = "Below target"

    queue = [row for row in rows if row["qualifies"]]
    if strategy == "highest_first":
        queue.sort(key=lambda row: (-_safe_int(row.get("trophies", 0), 0), row["brawler"]))
    elif strategy == "highest_winrate":
        queue.sort(key=lambda row: (-_safe_int(row.get("winrate", 50), 50), _safe_int(row.get("trophies", 0), 0), row["brawler"]))
    elif strategy == "manual_priority":
        queue.sort(key=lambda row: (_safe_int(row.get("priority", 0), 0), _safe_int(row.get("trophies", 0), 0), row["brawler"]))
    elif strategy == "alphabetical":
        queue.sort(key=lambda row: row["brawler"])
    else:
        queue.sort(key=lambda row: (_safe_int(row.get("trophies", 0), 0), row["brawler"]))

    for index, row in enumerate(queue, start=1):
        row["order"] = index

    if rows and not queue:
        if all(not row.get("owned", False) for row in rows):
            empty_reason = "No brawlers selected or unlocked."
        elif all(row.get("excluded") for row in rows if row.get("owned", False)):
            empty_reason = "All selected/unlocked brawlers are excluded."
        elif all(_safe_int(row.get("trophies", 0), 0) >= target for row in rows if row.get("owned", False) and row.get("included", True) and not row.get("excluded")):
            empty_reason = "All selected/unlocked brawlers are already at or above target."
        else:
            empty_reason = "No brawlers qualify for the current farm settings."
    elif not rows and mode == "api":
        empty_reason = "API roster not loaded. Fetch your roster from the Brawl Stars API."
    elif not rows:
        empty_reason = "No brawlers loaded."
    else:
        empty_reason = ""

    return {
        "mode": mode,
        "strategy": strategy,
        "target": target,
        "rows": rows,
        "queue": queue,
        "empty_reason": empty_reason,
    }
