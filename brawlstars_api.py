import requests


API_BASE_URL = "https://api.brawlstars.com/v1"


def normalize_player_tag(player_tag):
    cleaned = str(player_tag).strip().upper().replace(" ", "")
    if not cleaned:
        raise ValueError("Enter a Brawl Stars player tag first.")
    if cleaned.startswith("#"):
        cleaned = cleaned[1:]
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

    payload = response.json()
    brawler_data = {}
    for entry in payload.get("brawlers", []):
        normalized_name = normalize_brawler_name(entry.get("name", ""))
        if not normalized_name:
            continue
        brawler_data[normalized_name] = {
            "name": entry.get("name", normalized_name),
            "trophies": int(entry.get("trophies", 0)),
            "highestTrophies": int(entry.get("highestTrophies", entry.get("trophies", 0))),
            "power": int(entry.get("power", 0)),
            "rank": int(entry.get("rank", 0)),
        }

    return {
        "player_name": payload.get("name", ""),
        "player_tag": f"#{normalized_tag}",
        "brawlers": brawler_data,
    }
