import time
import cv2
import numpy as np


class StatsDashboard:
    """
    Live stats dashboard rendered as an OpenCV window.
    Shows trophies, match history, session info, gameplay status, etc.
    Refreshes at ~2 FPS to minimize CPU overhead.
    """

    WINDOW_NAME = "PylaAI Stats"
    WIDTH = 460  # Wider for BT/RL info
    BG_COLOR = (30, 30, 30)
    SECTION_COLOR = (50, 50, 50)
    HEADER_COLOR = (0, 140, 255)       # Orange
    LABEL_COLOR = (180, 180, 180)      # Light gray
    VALUE_COLOR = (255, 255, 255)      # White
    GREEN = (0, 220, 100)
    RED = (80, 80, 255)
    YELLOW = (0, 220, 220)
    CYAN = (220, 180, 0)
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self):
        self._last_update = 0
        self._update_interval = 0.5  # seconds between redraws
        self._ips = 0.0
        self._start_time = time.time()
        self._window_created = False

    def update(self, trophy_observer=None, play=None, stage_manager=None,
               state=None, ips=0.0, start_time=None, run_for_minutes=0,
               in_cooldown=False, brawler_data=None,
               bt_info=None, rl_metrics=None):
        """
        Called every frame from the main loop. Throttles actual redraws.
        """
        now = time.time()
        if now - self._last_update < self._update_interval:
            return
        self._last_update = now
        self._ips = ips
        if start_time:
            self._start_time = start_time

        try:
            img = self._render(
                trophy_observer=trophy_observer,
                play=play,
                stage_manager=stage_manager,
                state=state or "unknown",
                ips=ips,
                run_for_minutes=run_for_minutes,
                in_cooldown=in_cooldown,
                brawler_data=brawler_data,
                bt_info=bt_info,
                rl_metrics=rl_metrics,
            )
            if not self._window_created:
                cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
                cv2.moveWindow(self.WINDOW_NAME, 50, 50)
                self._window_created = True
            cv2.imshow(self.WINDOW_NAME, img)
            cv2.waitKey(1)
        except Exception:
            pass  # Never crash the bot

    def _render(self, trophy_observer, play, stage_manager, state, ips,
                run_for_minutes, in_cooldown, brawler_data, bt_info=None, rl_metrics=None):
        """Build the stats image."""
        lines = []  # list of (type, text, color) or (type, label, value, label_color, value_color)

        # sESSION
        lines.append(("header", "SESSION"))
        elapsed = time.time() - self._start_time
        lines.append(("kv", "Elapsed", self._fmt_time(elapsed)))
        lines.append(("kv", "IPS", f"{ips:.1f}"))
        state_color = self.GREEN if state == "match" else self.YELLOW
        lines.append(("kv_c", "State", state.upper(), self.LABEL_COLOR, state_color))
        if run_for_minutes > 0:
            remaining = max(0, run_for_minutes * 60 - elapsed)
            lines.append(("kv", "Timer", self._fmt_time(remaining) + " left"))
        else:
            lines.append(("kv", "Timer", "unlimited"))
        if in_cooldown:
            lines.append(("kv_c", "Cooldown", "ACTIVE", self.LABEL_COLOR, self.RED))

        # bRAWLER
        brawler_name = "?"
        target = "?"
        current_val = "?"
        push_type = "trophies"
        if brawler_data and len(brawler_data) > 0:
            bd = brawler_data[0]
            brawler_name = bd.get("brawler", "?")
            push_type = bd.get("type", "trophies")
            target = bd.get("push_until", "?")

        lines.append(("header", f"BRAWLER: {brawler_name.upper()}"))

        if trophy_observer:
            trophies = trophy_observer.current_trophies
            wins = trophy_observer.current_wins
            win_streak = trophy_observer.win_streak
            matches = trophy_observer.match_counter

            if trophies is not None:
                lines.append(("kv", "Trophies", f"{trophies}  (target: {target})"))
            if wins is not None:
                lines.append(("kv", "Wins", str(wins)))
            lines.append(("kv", "Win Streak", str(win_streak)))
            lines.append(("kv", "Matches", str(matches)))

            # mATCH HISTORY
            lines.append(("header", "MATCH HISTORY"))
            hist = trophy_observer.match_history.get(brawler_name, {})
            v = hist.get("victory", 0)
            d = hist.get("defeat", 0)
            dr = hist.get("draw", 0)
            total_games = v + d + dr
            if total_games > 0:
                vp = v / total_games * 100
                dp = d / total_games * 100
                drp = dr / total_games * 100
                lines.append(("kv_c", "Victories", f"{v}  ({vp:.0f}%)", self.LABEL_COLOR, self.GREEN))
                lines.append(("kv_c", "Defeats", f"{d}  ({dp:.0f}%)", self.LABEL_COLOR, self.RED))
                lines.append(("kv_c", "Draws", f"{dr}  ({drp:.0f}%)", self.LABEL_COLOR, self.YELLOW))
            else:
                lines.append(("kv", "No matches yet", ""))

            # Total across all brawlers
            total_hist = trophy_observer.match_history.get("total", {})
            tv = total_hist.get("victory", 0)
            td = total_hist.get("defeat", 0)
            tdr = total_hist.get("draw", 0)
            total_all = tv + td + tdr
            if total_all > 0:
                lines.append(("kv", "Total (all)", f"{tv}W / {td}L / {tdr}D"))
        else:
            lines.append(("kv", "Waiting for data...", ""))

        # gAMEPLAY
        lines.append(("header", "GAMEPLAY"))
        if play:
            movement = getattr(play, 'last_movement', '?')
            lines.append(("kv", "Movement", str(movement) if movement else "idle"))

            gadget = getattr(play, 'is_gadget_ready', False)
            super_ready = getattr(play, 'is_super_ready', False)
            hyper = getattr(play, 'is_hypercharge_ready', False)
            lines.append(("kv_c", "Gadget", "READY" if gadget else "not ready",
                           self.LABEL_COLOR, self.GREEN if gadget else self.LABEL_COLOR))
            lines.append(("kv_c", "Super", "READY" if super_ready else "not ready",
                           self.LABEL_COLOR, self.GREEN if super_ready else self.LABEL_COLOR))
            lines.append(("kv_c", "Hypercharge", "READY" if hyper else "not ready",
                           self.LABEL_COLOR, self.GREEN if hyper else self.LABEL_COLOR))

            stuck = getattr(play, 'fix_movement_keys', {}).get('toggled', False)
            if stuck:
                lines.append(("kv_c", "Anti-stuck", "ACTIVE", self.LABEL_COLOR, self.RED))

            # Detection info
            lines.append(("header", "DETECTION"))
            t_player = getattr(play, 'time_since_player_last_found', 0)
            if t_player > 0:
                ago = time.time() - t_player
                if ago < 1:
                    lines.append(("kv_c", "Player", "FOUND", self.LABEL_COLOR, self.GREEN))
                else:
                    lines.append(("kv_c", "Player", f"lost ({ago:.0f}s ago)", self.LABEL_COLOR, self.RED))
            else:
                lines.append(("kv", "Player", "waiting..."))

            t_det = getattr(play, 'time_since_detections', {})
            if 'enemy' in t_det:
                enemy_ago = time.time() - t_det['enemy']
                if enemy_ago < 2:
                    lines.append(("kv_c", "Enemy", "VISIBLE", self.LABEL_COLOR, self.RED))
                else:
                    lines.append(("kv", "Enemy", f"last seen {enemy_ago:.0f}s ago"))

            walls = len(getattr(play, 'last_walls_data', []))
            lines.append(("kv", "Walls tracked", str(walls)))
        else:
            lines.append(("kv", "Waiting...", ""))

        # bEHAVIOR TREE AI
        if bt_info:
            lines.append(("header", "BEHAVIOR TREE"))
            bt_active = bt_info.get("active", False)
            lines.append(("kv_c", "Mode", "BT ACTIVE" if bt_active else "RULES",
                          self.LABEL_COLOR, self.GREEN if bt_active else self.LABEL_COLOR))
            if bt_active:
                # Current tree path
                path = bt_info.get("tree_path", [])
                if path:
                    path_str = " -> ".join(path[:3])  # First 3 nodes
                    if len(path) > 3:
                        path_str += "..."
                    lines.append(("kv", "Path", path_str))
                # Tick count
                ticks = bt_info.get("tick_count", 0)
                lines.append(("kv", "Ticks", str(ticks)))
                # Decision reason
                reason = bt_info.get("reason", "")
                if reason:
                    lines.append(("kv", "Reason", reason[:25]))
                # Subsystem status
                subs = bt_info.get("subsystems", {})
                active_subs = sum(1 for v in subs.values() if v)
                total_subs = len(subs)
                lines.append(("kv_c", "Subsystems", f"{active_subs}/{total_subs} active",
                              self.LABEL_COLOR, self.GREEN if active_subs == total_subs else self.YELLOW))
                # Combo status
                if bt_info.get("combo_active"):
                    lines.append(("kv_c", "Combo", "EXECUTING", self.LABEL_COLOR, self.CYAN))

        # rEINFORCEMENT LEARNING
        if rl_metrics:
            lines.append(("header", "RL METRICS"))
            # Episode reward
            ep_reward = rl_metrics.get("episode_reward", 0.0)
            reward_color = self.GREEN if ep_reward > 0 else (self.RED if ep_reward < 0 else self.VALUE_COLOR)
            lines.append(("kv_c", "Ep Reward", f"{ep_reward:+.1f}", self.LABEL_COLOR, reward_color))
            # Kills / deaths
            kills = rl_metrics.get("kills", 0)
            deaths = rl_metrics.get("deaths", 0)
            lines.append(("kv", "K/D", f"{kills} / {deaths}"))
            # Damage dealt / taken
            dealt = rl_metrics.get("damage_dealt", 0)
            taken = rl_metrics.get("damage_taken", 0)
            lines.append(("kv", "Dmg", f"+{dealt} / -{taken}"))
            # Accuracy
            acc = rl_metrics.get("hit_rate", -1)
            if acc >= 0:
                acc_color = self.GREEN if acc > 0.5 else self.YELLOW
                lines.append(("kv_c", "Accuracy", f"{acc*100:.0f}%", self.LABEL_COLOR, acc_color))
            # Opponent profiles
            n_profiles = rl_metrics.get("opponent_profiles", 0)
            if n_profiles > 0:
                lines.append(("kv", "Opp Profiles", str(n_profiles)))
            # Training mode
            if rl_metrics.get("training_enabled"):
                lines.append(("kv_c", "Training", "LEARNING", self.LABEL_COLOR, self.CYAN))

        # bRAWLER QUEUE
        if brawler_data and len(brawler_data) > 1:
            lines.append(("header", "QUEUE"))
            for i, bd in enumerate(brawler_data[1:], 1):
                lines.append(("kv", f"  #{i+1}", f"{bd['brawler']} -> {bd.get('push_until', '?')}"))

        # render
        return self._draw_lines(lines)

    def _draw_lines(self, lines):
        """Render the list of lines into an image."""
        # Calculate height
        line_height = 22
        header_height = 30
        padding = 8
        total_h = padding
        for entry in lines:
            if entry[0] == "header":
                total_h += header_height + 4
            else:
                total_h += line_height
        total_h += padding * 2

        img = np.full((max(total_h, 200), self.WIDTH, 3), self.BG_COLOR, dtype=np.uint8)

        y = padding
        for entry in lines:
            if entry[0] == "header":
                # Draw section header with background bar
                y += 4
                cv2.rectangle(img, (0, y), (self.WIDTH, y + header_height), self.SECTION_COLOR, -1)
                cv2.putText(img, entry[1], (10, y + 20), self.FONT, 0.55, self.HEADER_COLOR, 1, cv2.LINE_AA)
                y += header_height
            elif entry[0] == "kv":
                # Key-value pair
                label, value = entry[1], entry[2]
                cv2.putText(img, f"{label}:", (15, y + 16), self.FONT, 0.42, self.LABEL_COLOR, 1, cv2.LINE_AA)
                cv2.putText(img, str(value), (180, y + 16), self.FONT, 0.42, self.VALUE_COLOR, 1, cv2.LINE_AA)
                y += line_height
            elif entry[0] == "kv_c":
                # Key-value with custom colors
                label, value = entry[1], entry[2]
                lbl_color = entry[3] if len(entry) > 3 else self.LABEL_COLOR
                val_color = entry[4] if len(entry) > 4 else self.VALUE_COLOR
                cv2.putText(img, f"{label}:", (15, y + 16), self.FONT, 0.42, lbl_color, 1, cv2.LINE_AA)
                cv2.putText(img, str(value), (180, y + 16), self.FONT, 0.42, val_color, 1, cv2.LINE_AA)
                y += line_height

        # Bottom border line
        cv2.line(img, (0, y + 4), (self.WIDTH, y + 4), self.SECTION_COLOR, 1)

        return img

    @staticmethod
    def _fmt_time(seconds):
        """Format seconds as HH:MM:SS."""
        s = int(seconds)
        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def destroy(self):
        """Close the stats window."""
        try:
            cv2.destroyWindow(self.WINDOW_NAME)
        except Exception:
            pass

    @staticmethod
    def extract_bt_info(play):
        """Extract BT info from play instance for dashboard display."""
        if play is None:
            return None
        bt = getattr(play, '_bt_combat', None)
        if bt is None:
            return {"active": False}

        try:
            debug = bt.get_debug_info() if hasattr(bt, 'get_debug_info') else {}
            return {
                "active": True,
                "tree_path": debug.get("tree_path", []),
                "tick_count": debug.get("tick_count", 0),
                "reason": debug.get("reason", ""),
                "combo_active": debug.get("combo_active", False),
                "subsystems": {
                    "enemy_tracker": bt._enemy_tracker is not None,
                    "projectile_detector": bt._projectile_detector is not None,
                    "spatial_memory": bt._spatial_memory is not None,
                    "combo_engine": bt._combo_engine is not None,
                    "opponent_model": bt._opponent_model is not None,
                    "manual_aimer": bt._manual_aimer is not None,
                    "ammo_reader": bt._ammo_reader is not None,
                    "hp_estimator": bt._hp_estimator is not None,
                }
            }
        except Exception:
            return {"active": True, "error": True}

    @staticmethod
    def extract_rl_metrics(play):
        """Extract RL metrics from play instance for dashboard display."""
        if play is None:
            return None
        bt = getattr(play, '_bt_combat', None)
        if bt is None:
            return None

        try:
            # Get reward calculator from play or bt
            rc = getattr(bt, '_reward_calculator', None)
            om = getattr(bt, '_opponent_model', None)

            metrics = {}

            if rc:
                summary = rc.episode_summary() if hasattr(rc, 'episode_summary') else {}
                metrics["episode_reward"] = summary.get("total_reward", 0.0)
                metrics["kills"] = summary.get("kills", 0)
                metrics["deaths"] = summary.get("deaths", 0)
                metrics["damage_dealt"] = int(summary.get("damage_dealt", 0))
                metrics["damage_taken"] = int(summary.get("damage_taken", 0))
                metrics["hit_rate"] = summary.get("hit_rate", -1)

            if om:
                metrics["opponent_profiles"] = len(getattr(om, 'profiles', {}))

            # Training status (if trainer exists)
            trainer = getattr(play, '_rl_trainer', None)
            metrics["training_enabled"] = trainer is not None

            return metrics if metrics else None
        except Exception:
            return None
