import requests
import re


API_BASE_URL = "https://api.brawlstars.com/v1"


def normalize_player_tag(player_tag):
    cleaned = str(player_tag).strip().upper().replace(" ", "")
    if not cleaned:
        raise ValueError("Enter a Brawl Stars player tag first.")
    if cleaned.startswith("#"):
        cleaned = cleaned[1:]
    if not re.fullmatch(r"[A-Z0-9]{3,15}", cleaned):
        raise ValueError("Enter a valid Brawl Stars player tag using letters and numbers only.")
    return cleaned


def normalize_brawler_name(name):
    cleaned = (
        str(name)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("-", "")
        .replace(".", "")
        .replace("&", "")
        .replace("'", "")
        .replace("/", "")
    )
    alias_map = {
        "8bit": "8bit",
        "elprimo": "elprimo",
        "mrp": "mrp",
        "rt": "rt",
        "larrylawrie": "larrylawrie",
    }
    return alias_map.get(cleaned, cleaned)


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def _names_from_api_list(values):
    names = []
    for item in values or []:
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            if name:
                names.append(name)
    return names


def parse_player_profile_payload(payload, player_tag=""):
    if not isinstance(payload, dict):
        raise ValueError("The Brawl Stars API returned a malformed player profile.")

    brawler_data = {}
    for entry in payload.get("brawlers", []) or []:
        if not isinstance(entry, dict):
            continue
        normalized_name = normalize_brawler_name(entry.get("name", ""))
        if not normalized_name:
            continue
        brawler_data[normalized_name] = {
            "name": entry.get("name", normalized_name),
            "trophies": _safe_int(entry.get("trophies"), 0),
            "highestTrophies": _safe_int(entry.get("highestTrophies", entry.get("trophies", 0)), 0),
            "power": _safe_int(entry.get("power"), 0),
            "rank": _safe_int(entry.get("rank"), 0),
            "gears": _names_from_api_list(entry.get("gears")),
            "gadgets": _names_from_api_list(entry.get("gadgets")),
            "starPowers": _names_from_api_list(entry.get("starPowers")),
            "source": "api",
            "owned": True,
            "included": True,
        }

    normalized_tag = ""
    if player_tag:
        normalized_tag = f"#{normalize_player_tag(player_tag)}"
    elif payload.get("tag"):
        normalized_tag = str(payload.get("tag", "")).strip().upper()

    return {
        "player_name": payload.get("name", ""),
        "player_tag": normalized_tag,
        "brawlers": brawler_data,
    }


def fetch_player_profile(api_key, player_tag, timeout=15):
    api_key = str(api_key).strip()
    if not api_key:
        raise ValueError("Enter your Brawl Stars API key first.")

    normalized_tag = normalize_player_tag(player_tag)
    encoded_tag = f"%23{normalized_tag}"
    response = requests.get(
        f"{API_BASE_URL}/players/{encoded_tag}",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )

    if response.status_code in (401, 403):
        raise ValueError("The Brawl Stars API key was rejected.")
    if response.status_code == 404:
        raise ValueError("The player tag was not found.")
    if response.status_code != 200:
        try:
            error_message = response.json().get("message", response.text)
        except Exception:
            error_message = response.text
        raise RuntimeError(f"Brawl Stars API request failed: {error_message}")

    return parse_player_profile_payload(response.json(), normalized_tag)
