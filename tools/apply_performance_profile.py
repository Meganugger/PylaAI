import argparse
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from performance_profile import apply_performance_profile, get_performance_profile_summary


def main():
    parser = argparse.ArgumentParser(
        description="Apply known-good PylaAI performance settings without editing emulator internals."
    )
    parser.add_argument(
        "--profile",
        choices=["balanced", "low-end", "low_end", "quality"],
        default="balanced",
        help="Performance profile to apply.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be applied without saving.")
    args = parser.parse_args()

    print("PylaAI performance profile")
    print(f"Python: {platform.python_version()} {platform.architecture()[0]}")
    if platform.architecture()[0] != "64bit":
        print("WARNING: 32-bit Python is not supported. Use a 64-bit Python runtime.")

    print(get_performance_profile_summary(args.profile))
    result = apply_performance_profile(args.profile, save=not args.dry_run)

    action = "Would apply" if args.dry_run else "Applied"
    print(f"{action} profile: {result['profile']}")
    print("Updated cfg/general_config.toml keys:")
    for key in result["changed_general_keys"]:
        print(f"- {key} = {result['general_config'][key]}")
    print("Updated cfg/bot_config.toml keys:")
    for key in result["changed_bot_keys"]:
        print(f"- {key} = {result['bot_config'][key]}")

    if not args.dry_run:
        print("")
        print("Restart the bot after applying this profile.")
    print("Keep emulator graphics at 1920x1080 landscape and avoid emulator eco or low-FPS mode.")


if __name__ == "__main__":
    main()
