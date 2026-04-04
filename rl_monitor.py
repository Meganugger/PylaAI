"""RL training monitor -- run to check learning progress."""
import os, json, time, argparse, sys

RL_DIR = "rl_models"
POLICY_FILE = os.path.join(RL_DIR, "policy.pt")
STATS_FILE = os.path.join(RL_DIR, "training_stats.json")


def _status(ok: bool | None, warn: bool = False) -> str:
    if ok is None:
        return "[GRAY] N/A"
    if ok:
        return "[GREEN] OK"
    if warn:
        return "[YELLOW] WARN"
    return "[RED] FAIL"


def _thresholds(mode: str, episodes: int) -> dict:
    if mode == "auto":
        mode = "lenient" if episodes < 3000 else "strict"

    if mode == "strict":
        return {
            "kd_min": 0.7,
            "kd_max": 1.2,
            "death_rate_max": 0.60,
            "trade_min": 1.0,
        }

    # lenient
    return {
        "kd_min": 0.55,
        "kd_max": 1.30,
        "death_rate_max": 0.70,
        "trade_min": 0.85,
    }


def _classify_verification(verification: dict, thresholds: dict, episodes: int) -> dict:
    kd20 = verification.get("recent_kd_20", 0.0)
    death_rate20 = verification.get("death_rate_20", 1.0)
    trade20 = verification.get("damage_trade_ratio_20", 0.0)
    onek_oned = verification.get("one_k_one_d_episode_negative", None)
    equal_trade_neg = verification.get("equal_damage_trade_negative", None)
    value_stable = verification.get("value_loss_stable", None)
    passive_hide = verification.get("passive_hide_risk", None)

    enough_recent_window = episodes >= 20
    one_k_one_d_samples = int(verification.get("one_k_one_d_samples", 0) or 0)
    equal_trade_samples = int(verification.get("equal_trade_samples", 0) or 0)

    return {
        "kd20": ((kd20 >= thresholds["kd_min"] and kd20 <= thresholds["kd_max"]) if enough_recent_window else None),
        "death_rate20": ((death_rate20 <= thresholds["death_rate_max"]) if enough_recent_window else None),
        "trade20": ((trade20 >= thresholds["trade_min"]) if enough_recent_window else None),
        "onek_oned": (onek_oned if one_k_one_d_samples >= 5 else None),
        "equal_trade_neg": (equal_trade_neg if equal_trade_samples >= 5 else None),
        "value_stable": value_stable,
        "passive_hide": (False if passive_hide is True else (True if passive_hide is False else None)),
    }


def _overall_verdict(verdicts: dict) -> tuple[str, str]:
    hard_keys = ["kd20", "trade20", "value_stable", "passive_hide"]
    soft_keys = ["death_rate20", "onek_oned", "equal_trade_neg"]

    hard_vals = [verdicts.get(k) for k in hard_keys if verdicts.get(k) is not None]
    soft_vals = [verdicts.get(k) for k in soft_keys if verdicts.get(k) is not None]

    if not hard_vals and not soft_vals:
        return "INSUFFICIENT_DATA", "[GRAY]"

    if hard_vals and all(hard_vals) and all(v is not False for v in soft_vals):
        return "PASS", "[GREEN]"

    hard_fail_count = sum(1 for v in hard_vals if v is False)
    soft_fail_count = sum(1 for v in soft_vals if v is False)

    if hard_fail_count >= 2 or (hard_fail_count >= 1 and soft_fail_count >= 1):
        return "FAIL", "[RED]"

    return "WARN", "[YELLOW]"


def _build_report(mode: str = "auto") -> dict:
    now = time.strftime('%H:%M:%S')
    report = {
        "timestamp": now,
        "mode_requested": mode,
        "active_mode": mode,
        "status": "ok",
        "paths": {
            "rl_dir": RL_DIR,
            "policy_file": POLICY_FILE,
            "stats_file": STATS_FILE,
        },
        "policy": {
            "exists": os.path.exists(POLICY_FILE),
            "size_bytes": 0,
            "last_save_time": None,
        },
        "stats": {
            "exists": os.path.exists(STATS_FILE),
            "episodes": 0,
            "updates": 0,
            "rewards": {},
            "win_loss": {},
            "combat": {},
            "verification": {},
            "reward_usage": {},
            "loss": {},
            "recent_matches": {},
        },
        "messages": [],
    }

    if not os.path.exists(RL_DIR):
        report["status"] = "no_rl_dir"
        report["messages"].append("rl_models_missing")
        return report

    if report["policy"]["exists"]:
        report["policy"]["size_bytes"] = os.path.getsize(POLICY_FILE)
        report["policy"]["last_save_time"] = time.strftime('%H:%M:%S', time.localtime(os.path.getmtime(POLICY_FILE)))

    if not report["stats"]["exists"]:
        report["status"] = "no_stats_file"
        report["messages"].append("training_stats_missing")
        return report

    with open(STATS_FILE, "r") as f:
        stats = json.load(f)

    overview = stats.get("overview", {})
    rewards = stats.get("rewards", {})
    win_loss = stats.get("win_loss", {})
    combat = stats.get("combat", {})
    verification = stats.get("verification", {})
    reward_usage = stats.get("reward_usage", {})
    recent_matches = stats.get("recent_matches", [])
    loss_history = stats.get("loss_history", [])

    episodes = int(overview.get("total_episodes", 0))
    total_updates = int(overview.get("total_updates", 0))
    thresholds = _thresholds(mode, episodes)
    active_mode = mode if mode != "auto" else ("lenient" if episodes < 3000 else "strict")

    report["active_mode"] = active_mode
    report["stats"]["episodes"] = episodes
    report["stats"]["updates"] = total_updates
    report["stats"]["rewards"] = {
        "avg_reward_last_10": float(rewards.get("avg_reward_last_10", 0.0)),
        "avg_reward_last_50": float(rewards.get("avg_reward_last_50", 0.0)),
        "best_reward": float(rewards.get("best_reward", 0.0)),
        "trend": rewards.get("trend", "unknown"),
        "trend_delta": float(rewards.get("trend_delta", 0.0)),
    }
    report["stats"]["win_loss"] = {
        "global_win_rate": win_loss.get("global_win_rate", "0.0%"),
        "recent_win_rate_20": win_loss.get("recent_win_rate_20", "0.0%"),
    }
    report["stats"]["combat"] = {
        "total_kills": combat.get("total_kills", 0),
        "total_deaths": combat.get("total_deaths", 0),
        "global_kda_per_match": combat.get("global_kda_per_match", 0),
    }

    if verification:
        verdicts = _classify_verification(verification, thresholds, episodes)
        overall, _tag = _overall_verdict(verdicts)
        report["stats"]["verification"] = {
            "thresholds": thresholds,
            "metrics": verification,
            "verdicts": verdicts,
            "overall": overall,
            "insufficient_recent_window": episodes < 20,
            "minimum_samples": {
                "one_k_one_d": 5,
                "equal_trade": 5,
            },
        }

    top_usage = reward_usage.get("top_usage_rates", [])
    report["stats"]["reward_usage"] = {
        "top_usage_rates": top_usage,
        "disabled_signals": reward_usage.get("disabled_signals", {}),
    }

    if loss_history:
        last3 = [row.get("value_loss", 0.0) for row in loss_history[-3:]]
        loss_info = {
            "value_loss_last3": [round(v, 4) for v in last3],
            "trend": "insufficient_data",
        }
        if len(loss_history) > 5:
            vals = [row.get("value_loss", 0.0) for row in loss_history if isinstance(row, dict)]
            early = sum(vals[:3]) / max(1, len(vals[:3]))
            late = sum(vals[-3:]) / max(1, len(vals[-3:]))
            if late < early * 0.8:
                loss_info["trend"] = "dropping"
            elif late > early * 1.2:
                loss_info["trend"] = "rising"
            else:
                loss_info["trend"] = "stable"
        report["stats"]["loss"] = loss_info

    if recent_matches:
        last5 = recent_matches[-5:]
        report["stats"]["recent_matches"] = {
            "count": len(last5),
            "last5_rewards": [m.get("reward", 0) for m in last5],
        }

    if episodes >= 10:
        report["status"] = "active_training" if total_updates > 0 else "episodes_no_updates"
    elif episodes > 0:
        report["status"] = "collecting_data"

    return report


def _exit_code_from_report(
    report: dict,
    fail_on_warn: bool = False,
    fail_on_insufficient_data: bool = False,
) -> int:
    verification = report.get("stats", {}).get("verification", {})
    overall = verification.get("overall")
    if fail_on_insufficient_data and overall == "INSUFFICIENT_DATA":
        return 1
    if fail_on_warn and overall == "WARN":
        return 1
    return 1 if overall == "FAIL" else 0

def check(mode: str = "auto", json_output: bool = False):
    report = _build_report(mode=mode)

    if json_output:
        print(json.dumps(report, indent=2))
        return report

    print("=" * 60)
    print(f"  RL TRAINING MONITOR  --  {time.strftime('%H:%M:%S')}")
    print("=" * 60)

    if report["status"] == "no_rl_dir":
        print("[!] rl_models/ does not exist yet -- no training so far.")
        print("    Start a match and wait for it to end.")
        return report

    # policy file
    if report["policy"]["exists"]:
        size = report["policy"]["size_bytes"]
        mtime = report["policy"]["last_save_time"]
        print(f"[+] policy.pt: {size:,} bytes (last save: {mtime})")
    else:
        print("[-] policy.pt not saved yet (saves every 10 episodes)")

    # training stats
    if report["stats"]["exists"]:
        episodes = report["stats"]["episodes"]
        total_updates = report["stats"]["updates"]
        active_mode = report["active_mode"]
        rewards = report["stats"]["rewards"]
        win_loss = report["stats"]["win_loss"]
        combat = report["stats"]["combat"]
        verification_data = report["stats"]["verification"]
        reward_usage = report["stats"]["reward_usage"]
        loss_data = report["stats"]["loss"]
        recent_matches_data = report["stats"]["recent_matches"]

        avg_reward_10 = rewards.get("avg_reward_last_10", 0.0)
        avg_reward_50 = rewards.get("avg_reward_last_50", 0.0)
        best_reward = rewards.get("best_reward", 0.0)
        trend = rewards.get("trend", "unknown")
        trend_delta = rewards.get("trend_delta", 0.0)

        print(f"\n[EPISODES]  {episodes} played")
        print(f"[UPDATES]   {total_updates} PPO updates")
        print(f"[MODE]      {active_mode}")
        print(f"[REWARD]    avg10: {avg_reward_10:.2f} | avg50: {avg_reward_50:.2f} | best: {best_reward:.2f}")
        print(f"[TREND]     {trend} ({trend_delta:+.2f})")

        print(f"[WINRATE]   global={win_loss.get('global_win_rate', '0.0%')} recent20={win_loss.get('recent_win_rate_20', '0.0%')}")
        print(f"[COMBAT]    K={combat.get('total_kills', 0)} D={combat.get('total_deaths', 0)} KDA/m={combat.get('global_kda_per_match', 0)}")

        # Short-test verification metrics
        if verification_data:
            thresholds = verification_data.get("thresholds", {})
            verification = verification_data.get("metrics", {})
            verdicts = verification_data.get("verdicts", {})
            overall = verification_data.get("overall", "INSUFFICIENT_DATA")
            overall_tag = {
                "PASS": "[GREEN]",
                "WARN": "[YELLOW]",
                "FAIL": "[RED]",
                "INSUFFICIENT_DATA": "[GRAY]",
            }.get(overall, "[GRAY]")
            print("\n[VERIFICATION]")
            print(
                f"  KD(20): {verification.get('recent_kd_20', 0)} {_status(verdicts['kd20'])} "
                f"(target {thresholds['kd_min']:.2f}-{thresholds['kd_max']:.2f})"
            )
            print(
                f"  DeathRate(20): {verification.get('death_rate_20', 0)} {_status(verdicts['death_rate20'])} "
                f"(target <={thresholds['death_rate_max']:.2f})"
            )
            print(
                f"  Trade(20): {verification.get('damage_trade_ratio_20', 0)} {_status(verdicts['trade20'])} "
                f"(target >={thresholds['trade_min']:.2f})"
            )
            print(
                f"  1K/1D negative: {verification.get('one_k_one_d_episode_negative')} "
                f"(n={verification.get('one_k_one_d_samples', 0)}, min=5) {_status(verdicts['onek_oned'], warn=True)}"
            )
            print(
                f"  1:1 trade negative: {verification.get('equal_damage_trade_negative')} "
                f"(n={verification.get('equal_trade_samples', 0)}, min=5) {_status(verdicts['equal_trade_neg'], warn=True)}"
            )
            print(
                f"  value stable: {verification.get('value_loss_stable')} "
                f"| spikes20: {verification.get('value_loss_spike_count_20', 0)} {_status(verdicts['value_stable'])}"
            )
            print(
                f"  passive-hide risk: {verification.get('passive_hide_risk')} "
                f"{_status(verdicts['passive_hide'], warn=True)}"
            )
            print(f"  OVERALL: {overall_tag} {overall}")
            if verification_data.get("insufficient_recent_window", False):
                print("  note: KD/DeathRate/Trade checks activate after 20 episodes")

        # Reward signal usage
        top_usage = reward_usage.get("top_usage_rates", [])
        if top_usage:
            top_str = ", ".join([f"{row.get('signal')}={row.get('rate'):.4f}" for row in top_usage[:5]])
            print(f"[USAGE]     {top_str}")
        disabled = reward_usage.get("disabled_signals", {})
        if disabled:
            print(f"[DISABLED]  {', '.join(disabled.keys())}")

        if loss_data:
            print(f"[LOSS]      value last3: {loss_data.get('value_loss_last3', [])}")
            if loss_data.get("trend") == "dropping":
                print("            OK loss is dropping -- model is learning!")
            elif loss_data.get("trend") == "rising":
                print("            [!] loss is rising -- possible problem")
            elif loss_data.get("trend") == "stable":
                print("            ~ loss stable")

        if recent_matches_data:
            rew = recent_matches_data.get("last5_rewards", [])
            print(f"[LAST5]     rewards={rew}")

        if episodes >= 10:
            if total_updates > 0:
                print("\n[STATUS] OK model is actively training!")
            else:
                print("\n[STATUS] [!] episodes running but no training update?")
        elif episodes > 0:
            print(f"\n[STATUS] still collecting data... ({episodes}/10 for first save)")

    else:
        print("[-] training_stats.json not found yet")
        files = os.listdir(RL_DIR) if os.path.exists(RL_DIR) else []
        if files:
            print(f"    files present: {files}")
        else:
            print("    folder is empty -- training hasn't started yet")

    print("=" * 60)
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PylaAI RL monitor")
    parser.add_argument(
        "--mode",
        choices=["auto", "strict", "lenient"],
        default="auto",
        help="Threshold profile for verification checks",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON report",
    )
    parser.add_argument(
        "--exit-code",
        action="store_true",
        help="Return process exit code based on verification overall (FAIL=1, else 0)",
    )
    parser.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="With --exit-code, also return 1 when verification overall is WARN",
    )
    parser.add_argument(
        "--fail-on-insufficient-data",
        action="store_true",
        help="With --exit-code, also return 1 when verification overall is INSUFFICIENT_DATA",
    )
    args = parser.parse_args()
    report = check(mode=args.mode, json_output=args.json)
    if args.exit_code:
        sys.exit(
            _exit_code_from_report(
                report,
                fail_on_warn=args.fail_on_warn,
                fail_on_insufficient_data=args.fail_on_insufficient_data,
            )
        )
