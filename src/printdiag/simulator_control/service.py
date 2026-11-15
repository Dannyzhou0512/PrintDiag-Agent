"""Human-operated, simulator-only jog service.

P10-8B-1 uses the simulator firmware-aware protocol:
  GET  /agent/firmware-motion-state
  POST /agent/control/session
  POST /agent/control/jog
  GET  /agent/control/status
  POST /agent/control/release

Motion is allowed only when firmware absolute mode, homed/trusted axes,
logical/native/workspace consistency and ground-truth consistency are all
verified at runtime. No environment variable or test flag can bypass this.
"""

from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import threading
from time import monotonic, sleep
from uuid import uuid4

from printdiag.simulator_control.audit import ControlAudit
from printdiag.simulator_control.client import (
    ControlClientError, SimulatorControlClient,
)
from printdiag.simulator_control.config import (
    BOUNDS_MM, FEEDRATE_MM_MIN, IDLE_TIMEOUT_S, POSITION_TOLERANCE_MM, STEPS_MM,
    ControlConfig,
)
from printdiag.simulator_control.policies import AXES, parse_position

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AUDIT_DIR = ROOT / "runs" / "p10_8b_sim_control"

TERMINAL = {"COMPLETED", "FAILED", "UNKNOWN", "EXPIRED", "RELEASED"}
UPSTREAM_READY_CHECKS = (
    "firmware_state", "absolute_mode", "axes_homed", "axes_trusted",
    "logical_native_workspace", "ground_truth", "idle",
)


def _all_true(mapping: dict, keys=("x", "y", "z")) -> bool:
    return all(isinstance(mapping.get(key), bool) and mapping.get(key) for key in keys)


def _finite_position(mapping: dict) -> bool:
    try:
        return all(float(mapping[axis]) == float(mapping[axis]) for axis in AXES)
    except (KeyError, TypeError, ValueError):
        return False


class SimulatorControlService:
    def __init__(self, config: ControlConfig | None = None, *,
                 client: SimulatorControlClient | None = None,
                 audit: ControlAudit | None = None, audit_dir: Path | None = None,
                 completion_timeout_s: float = 12.0) -> None:
        self.config = config or ControlConfig.from_environment()
        self._client = client or SimulatorControlClient()
        self.audit = audit or ControlAudit(audit_dir or DEFAULT_AUDIT_DIR)
        self._armed = False
        self._armed_snapshot_id: str | None = None
        self._hold_id: int | None = None
        self._motion_lock = threading.Lock()
        self._last_jog_at = 0.0
        self._last_arm_at = None
        self._completion_timeout_s = completion_timeout_s

    # ------------------------------------------------------------------ state
    def _read_ready(self) -> dict:
        """Read firmware and kinematic state and derive readiness."""
        try:
            firmware = self._client.firmware_state()
            ground = self._client.state()
        except ControlClientError as exc:
            return {"available": False, "reason": exc.code, "errors": [exc.code],
                    "firmware": None, "ground_truth": None,
                    "position": None, "idle": None, "motion_ready": False,
                    "coordinate_semantics_verified": False,
                    "blocked_reason": exc.code}

        errors: list[str] = []
        firmware_ok = isinstance(firmware, dict) and firmware.get("available") is True
        if not firmware_ok:
            errors.append("FIRMWARE_STATE_UNAVAILABLE")
        ground_position = parse_position(ground)
        if ground_position is None:
            errors.append("GROUND_TRUTH_UNAVAILABLE")

        absolute = firmware.get("absolute_mode") is True if firmware_ok else False
        relative = firmware.get("axis_relative", {}) if firmware_ok else {}
        if not absolute or any(relative.get(axis) is True for axis in AXES):
            errors.append("RELATIVE_MODE_ACTIVE")
        homed = firmware.get("axes_homed", {}) if firmware_ok else {}
        trusted = firmware.get("axes_trusted", {}) if firmware_ok else {}
        if not _all_true(homed):
            errors.append("AXES_NOT_HOMED")
        if not _all_true(trusted):
            errors.append("AXES_NOT_TRUSTED")

        logical = firmware.get("logical_position", {}) if firmware_ok else {}
        native = firmware.get("native_position", {}) if firmware_ok else {}
        offset = firmware.get("workspace_offset", {}) if firmware_ok else {}
        coordinate_ok = (_finite_position(logical) and _finite_position(native)
                         and _finite_position(offset))
        if not coordinate_ok:
            errors.append("LOGICAL_NATIVE_WORKSPACE_UNAVAILABLE")
        elif any(abs(float(logical[axis]) - (float(native[axis]) + float(offset[axis]))) > 0.01
                 for axis in AXES):
            errors.append("LOGICAL_NATIVE_OFFSET_MISMATCH")
            coordinate_ok = False
        if coordinate_ok and ground_position is not None:
            if any(abs(float(native[axis]) - float(ground_position[axis])) > POSITION_TOLERANCE_MM
                   for axis in AXES):
                errors.append("GROUND_TRUTH_MISMATCH")
                coordinate_ok = False
        if coordinate_ok:
            for axis in AXES:
                low, high = BOUNDS_MM[axis]
                if not (low <= float(native[axis]) <= high):
                    errors.append("POSITION_OUT_OF_BOUNDS")
                    coordinate_ok = False
                    break

        busy = True
        idle_payload = None
        if firmware_ok:
            busy = bool(firmware.get("planner_busy") or firmware.get("commands_queued")
                        or firmware.get("rx_pending") or firmware.get("printing")
                        or firmware.get("moves_queued", 0))
            idle_payload = {"idle": not busy,
                            "moves_queued": int(firmware.get("moves_queued", 0) or 0),
                            "rx_pending": bool(firmware.get("rx_pending", False)),
                            "commands_queued": bool(firmware.get("commands_queued", False))}
            if busy:
                errors.append("DEVICE_BUSY")
            if firmware.get("firmware_running") is False:
                errors.append("FIRMWARE_NOT_RUNNING")

        coordinate_semantics = coordinate_ok and not any(
            error in errors for error in ("RELATIVE_MODE_ACTIVE", "LOGICAL_NATIVE_OFFSET_MISMATCH"))
        blocked = errors[0] if errors else None
        snapshot_payload = {
            "firmware": ({key: value for key, value in firmware.items()
                          if key not in {"observed_at", "state_generation"}}
                         if isinstance(firmware, dict) else None),
            "ground_truth": ground_position,
            "errors": errors,
        }
        snapshot_id = sha256(json.dumps(snapshot_payload, sort_keys=True,
                                        default=str).encode("utf-8")).hexdigest()[:20]
        return {
            "available": firmware_ok and ground_position is not None,
            "reason": blocked,
            "errors": errors,
            "firmware": firmware,
            "ground_truth": ground_position,
            "position": ground_position,
            "idle": idle_payload,
            "snapshot_id": snapshot_id,
            "motion_ready": firmware_ok and not errors,
            "coordinate_semantics_verified": coordinate_semantics,
            "blocked_reason": blocked,
        }

    def _suggest_jog(self, ready: dict) -> dict | None:
        """Read-only direction suggestion; never sends G-code."""
        firmware = ready.get("firmware") or {}
        position = firmware.get("native_position") if isinstance(firmware, dict) else None
        if not isinstance(position, dict) or not all(axis in position for axis in AXES):
            return None
        suggestions = {}
        for axis in AXES:
            low, high = BOUNDS_MM[axis]
            current = float(position[axis])
            room_min = current - low
            room_max = high - current
            suggestions[axis] = {
                "axis": axis,
                "direction": "+" if room_max >= room_min else "-",
                "step_mm": 0.1,
                "clearance_mm": round(max(room_min, room_max), 3),
            }
        # Prefer X for the first manual jog when it has room; otherwise use the
        # axis with the largest clearance. This function only reads state.
        preferred = next((axis for axis in AXES
                          if suggestions[axis]["clearance_mm"] >= 0.1), None)
        if preferred is None:
            preferred = max(AXES, key=lambda axis: suggestions[axis]["clearance_mm"])
        result = dict(suggestions[preferred])
        result["axes"] = suggestions
        result["reason"] = f"{preferred.upper()} 距边界余量较大，先试 0.1 mm"
        return result

    def status(self) -> dict:
        base = {
            "enabled": self.config.enabled, "armed": self._armed,
            "simulator_url": self.config.simulator_url, "bounds": self.config.bounds,
            "steps_mm": list(STEPS_MM), "feedrate_mm_min": self.config.feedrate_mm_min,
            "home_supported": False, "stop_supported": False,
            "temperature_control": False, "agent_control_allowed": False,
            "last_arm_utc": self._last_arm_at,
            "protocol": "firmware-session-v1",
        }
        if not self.config.enabled:
            return {**base, "available": False,
                    "reason": "PRINTDIAG_SIM_CONTROL_ENABLED is not 1",
                    "position": None, "idle": None, "position_trusted": False,
                    "motion_ready": False, "coordinate_semantics_verified": False,
                    "blocked_reason": "SIM_CONTROL_DISABLED", "firmware": None,
                    "ground_truth": None, "errors": ["SIM_CONTROL_DISABLED"]}
        ready = self._read_ready()
        try:
            control = self._client.control_status()
            session = control.get("session")
            hold = control.get("hold")
        except ControlClientError:
            session = None
            hold = None
        # Hold expiry invalidates the human arm; never leave a stale armed UI.
        if self._armed and not (isinstance(hold, dict) and hold.get("active") is True):
            self._armed = False
            self._armed_snapshot_id = None
            self._hold_id = None
            base["armed"] = False
        suggested = self._suggest_jog(ready)
        return {**base, **ready, "session": session, "hold": hold,
                "suggested_jog": suggested,
                "armed_snapshot_id": self._armed_snapshot_id,
                "hold_id": self._hold_id,
                "position_trusted": ready["motion_ready"],
                "homed_known": True,
                "warning": ("固件坐标模式/归零/工作坐标/ground truth 均一致，可人工点动。"
                            if ready["motion_ready"] else
                            "运行时条件未满足，真实点动被拒绝："
                            + (ready.get("blocked_reason") or "UNKNOWN"))}

    # -------------------------------------------------------------------- arm
    def arm(self, *, confirm_local_simulator: bool) -> dict:
        if not self.config.enabled:
            return {"ok": False, "code": "SIM_CONTROL_DISABLED"}
        if confirm_local_simulator is not True:
            return {"ok": False, "code": "CONFIRMATION_REQUIRED"}
        ready = self._read_ready()
        if not ready["motion_ready"]:
            return {"ok": False, "code": ready["blocked_reason"] or "MOTION_NOT_READY",
                    "errors": ready["errors"], "motion_ready": False}
        try:
            hold = self._client.hold_acquire()
        except ControlClientError as exc:
            return {"ok": False, "code": exc.code, "motion_ready": False}
        hold_payload = hold.get("hold") if isinstance(hold, dict) else None
        if not isinstance(hold_payload, dict) or hold_payload.get("active") is not True:
            return {"ok": False, "code": "HOLD_NOT_GRANTED", "motion_ready": False}
        self._hold_id = int(hold_payload.get("hold_id") or 0)
        self._armed = True
        self._armed_snapshot_id = ready.get("snapshot_id")
        self._last_arm_at = datetime.now(UTC).isoformat()
        self.audit.append({
            "request_id": uuid4().hex, "outcome": "armed", "source": "manual_web",
            "protocol": "firmware-session-v1", "firmware_session_id":
                (ready["firmware"] or {}).get("firmware_session_id"),
            "hold_id": self._hold_id,
            "hold_remaining_ms": hold_payload.get("remaining_ms"),
            "note": "operator confirmed local simulator after runtime readiness checks",
        })
        return {"ok": True, "armed": True, "motion_ready": True,
                "snapshot_id": self._armed_snapshot_id,
                "hold_id": self._hold_id,
                "hold": hold_payload,
                "firmware_session_id": (ready["firmware"] or {}).get("firmware_session_id"),
                "warning": "仅本地模拟器；会话/回执/位置复核仍会逐次检查。"}

    # -------------------------------------------------------------------- jog
    def disarm(self) -> dict:
        hold_id = self._hold_id
        self._armed = False
        self._armed_snapshot_id = None
        self._hold_id = None
        if hold_id:
            try:
                self._client.hold_release(hold_id)
            except ControlClientError as exc:
                return {"ok": False, "code": exc.code, "hold_id": hold_id}
        self.audit.append({"request_id": uuid4().hex, "outcome": "disarmed",
                           "source": "manual_web", "protocol": "firmware-session-v1",
                           "hold_id": hold_id})
        return {"ok": True, "armed": False, "hold_id": None}

    def renew_hold(self) -> dict:
        if not self._armed or not self._hold_id:
            return {"ok": False, "code": "CONTROL_NOT_ARMED"}
        try:
            result = self._client.hold_renew(self._hold_id)
        except ControlClientError as exc:
            self._armed = False
            self._hold_id = None
            return {"ok": False, "code": exc.code}
        self.audit.append({"request_id": uuid4().hex, "outcome": "hold_renewed",
                           "source": "manual_web", "protocol": "firmware-session-v1",
                           "hold_id": self._hold_id})
        return {"ok": True, "hold": result.get("hold"), "hold_id": self._hold_id}

    def jog(self, *, axis: str, direction: str, step: float) -> dict:
        request_id = uuid4().hex
        started = datetime.now(UTC).isoformat()
        if not self.config.enabled:
            return self._reject(request_id, started, "SIM_CONTROL_DISABLED", axis, direction, step)
        if not self._armed:
            return self._reject(request_id, started, "CONTROL_NOT_ARMED", axis, direction, step)
        if axis not in AXES:
            return self._reject(request_id, started, "UNKNOWN_AXIS", axis, direction, step)
        if direction not in ("+", "-"):
            return self._reject(request_id, started, "UNKNOWN_DIRECTION", axis, direction, step)
        if step not in STEPS_MM:
            return self._reject(request_id, started, "INVALID_STEP", axis, direction, step)
        if not self._motion_lock.acquire(blocking=False):
            return self._reject(request_id, started, "CONTROL_BUSY", axis, direction, step)
        try:
            if monotonic() - self._last_jog_at < 0.4:
                return self._reject(request_id, started, "DUPLICATE_REQUEST",
                                    axis, direction, step)
            ready = self._read_ready()
            if self._armed_snapshot_id and ready.get("snapshot_id") != self._armed_snapshot_id:
                return self._reject(request_id, started, "STATE_CHANGED_REARM",
                                    axis, direction, step, firmware=ready["firmware"])
            try:
                hold = self._client.control_status().get("hold") or {}
            except ControlClientError as exc:
                return self._reject(request_id, started, exc.code,
                                    axis, direction, step, firmware=ready["firmware"])
            if not ready["motion_ready"]:
                return self._reject(request_id, started,
                                    ready["blocked_reason"] or "MOTION_NOT_READY",
                                    axis, direction, step, firmware=ready["firmware"])
            if hold.get("active") is not True:
                return self._reject(request_id, started, "HOLD_INACTIVE",
                                    axis, direction, step, firmware=ready["firmware"])
            before = ready["firmware"]["native_position"]
            logical_before = ready["firmware"]["logical_position"]
            low, high = BOUNDS_MM[axis]
            target_edge = float(before[axis]) + (step if direction == "+" else -step)
            if not (low <= target_edge <= high):
                return self._reject(request_id, started, "TARGET_OUT_OF_BOUNDS",
                                    axis, direction, step, before=before)
            session = None
            try:
                session = self._client.session_acquire()
            except ControlClientError as exc:
                return self._reject(request_id, started, exc.code, axis, direction, step,
                                    before=before)
            session_id = (session.get("session_id")
                          or (session.get("session") or {}).get("session_id"))
            if not session_id:
                # The acquire may have succeeded but returned an older schema.
                # Release any live session before refusing the command.
                try:
                    live = (self._client.control_status().get("session") or {})
                    if live.get("active") and live.get("session_id"):
                        self._client.session_release(live["session_id"])
                except ControlClientError:
                    pass
                return self._reject(request_id, started, "SESSION_NOT_GRANTED",
                                    axis, direction, step, before=before)
            command = None
            target = None
            target_logical = None
            receipt_semantics = None
            final = {}
            released = False
            try:
                jog = self._client.control_jog(session_id, axis, direction == "+", step)
                command = jog.get("command")
                target = jog.get("target_native")
                target_logical = jog.get("target_logical")
                receipt_semantics = jog.get("receipt_semantics")
                final = self._wait_terminal()
            except ControlClientError as exc:
                self._release_quietly(session_id)
                return self._record(request_id, started, "FAILED", exc.code,
                                    axis, direction, step, before, None, command,
                                    receipt_semantics=receipt_semantics)
            finally:
                if session_id:
                    released = self._release_quietly(session_id)
                    if not released and not final:
                        final = {"session": {"state": "UNKNOWN"}}
            final_state = (final.get("session") or {}).get("state", "UNKNOWN")
            try:
                after_fw = self._client.firmware_state()
                after_ground = parse_position(self._client.state())
                after = after_fw.get("native_position") if isinstance(after_fw, dict) else None
            except ControlClientError:
                after = None
                after_ground = None
            actual_delta = None
            if isinstance(before, dict) and isinstance(after, dict):
                actual_delta = {axis_name: float(after[axis_name]) - float(before[axis_name])
                                for axis_name in AXES}
            if final_state == "COMPLETED":
                expected = step if direction == "+" else -step
                target_value = (float(target[axis]) if isinstance(target, dict)
                                else float(before[axis]) + expected)
                reached = (isinstance(after, dict)
                           and abs(float(after[axis]) - target_value) <= 0.25)
                moved = (actual_delta is not None
                         and actual_delta[axis] * expected > 0
                         and abs(actual_delta[axis] - expected) <= 0.25)
                ground_ok = (after_ground is not None and isinstance(after, dict)
                             and abs(float(after_ground[axis]) - float(after[axis])) <= 0.25)
                if not (reached and moved and ground_ok):
                    final_state = "FAILED"
                    error = "POSITION_MISMATCH"
                else:
                    error = None
            elif final_state in ("FAILED", "UNKNOWN", "EXPIRED"):
                error = final_state
            else:
                final_state = "UNKNOWN"
                error = "COMPLETION_TIMEOUT"

            rebaselined = False
            rebaseline_reason = None
            post_ready = {}
            post_status = {}
            if final_state == "COMPLETED" and error is None:
                try:
                    post_status = self._client.control_status()
                    post_ready = self._read_ready()
                except ControlClientError:
                    rebaseline_reason = "POST_STATE_UNAVAILABLE"
                else:
                    rebaselined, rebaseline_reason = self._try_rebaseline_after_success(
                        released=released, session_id=session_id, command=command,
                        final=final, ready=ready, post_ready=post_ready,
                        post_status=post_status, before=before, target=target,
                        axis=axis, direction=direction, step=step)
                    if rebaselined:
                        after = post_ready.get("firmware", {}).get("native_position", after)
                        after_ground = post_ready.get("ground_truth", after_ground)
            return self._record(request_id, started, final_state, error, axis, direction, step,
                                before, target, command, after=after, target_logical=target_logical,
                                delta=actual_delta, session=final.get("session"),
                                ground_after=after_ground, logical_before=logical_before,
                                receipt_semantics=receipt_semantics,
                                rebaselined=rebaselined,
                                post_snapshot_id=post_ready.get("snapshot_id"),
                                rebaseline_reason=rebaseline_reason,
                                session_released=released,
                                post_session_state=(post_status.get("session") or {}).get("state")
                                if post_status else None)
        finally:
            self._last_jog_at = monotonic()
            self._motion_lock.release()

    def _wait_terminal(self) -> dict:
        deadline = monotonic() + self._completion_timeout_s
        last = {}
        while monotonic() < deadline:
            last = self._client.control_status()
            state = (last.get("session") or {}).get("state")
            if state in TERMINAL:
                return last
            sleep(0.1)
        return last

    def _release_quietly(self, session_id: str) -> bool:
        try:
            self._client.session_release(session_id)
            return True
        except ControlClientError:
            return False

    def _try_rebaseline_after_success(
            self, *, released: bool, session_id: str, command: str | None,
            final: dict, ready: dict, post_ready: dict, post_status: dict,
            before: dict, target: dict | None, axis: str, direction: str,
            step: float) -> tuple[bool, str | None]:
        """Move the armed baseline only after a fully corroborated success.

        The caller holds _motion_lock.  Any failed condition leaves the old
        baseline untouched, so the next jog still requires a new arm.
        """
        if not self._armed or not self._armed_snapshot_id:
            return False, "CONTROL_NOT_ARMED"
        if not released or not session_id:
            return False, "SESSION_NOT_RELEASED"
        final_session = final.get("session") or {}
        if final_session.get("state") != "COMPLETED":
            return False, "SESSION_NOT_COMPLETED"
        if final_session.get("session_id") not in (None, "", session_id):
            return False, "SESSION_ID_MISMATCH"
        if final_session.get("accepted") is not True or final_session.get("command_sent") is not True:
            return False, "RECEIPT_NOT_ACCEPTED"
        if not final_session.get("command_id"):
            return False, "COMMAND_ID_MISSING"
        if not command:
            return False, "COMMAND_MISSING"

        post_fw = post_ready.get("firmware") or {}
        post_native = post_fw.get("native_position") or {}
        post_ground = post_ready.get("ground_truth") or {}
        if not all(axis_name in post_native for axis_name in AXES):
            return False, "POST_POSITION_MISSING"
        if not isinstance(target, dict) or not all(axis_name in target for axis_name in AXES):
            return False, "TARGET_MISSING"

        # The completed move must match the command on every axis, and the
        # other axes must not have moved underneath us.
        for axis_name in AXES:
            if abs(float(post_native[axis_name]) - float(target[axis_name])) > 0.25:
                return False, "POSITION_MISMATCH"
            if abs(float(post_ground[axis_name]) - float(post_native[axis_name])) > 0.25:
                return False, "GROUND_TRUTH_MISMATCH"
        expected = step if direction == "+" else -step
        delta = float(post_native[axis]) - float(before[axis])
        if abs(delta - expected) > 0.25:
            return False, "DELTA_MISMATCH"
        for axis_name in AXES:
            if axis_name != axis and abs(float(post_native[axis_name]) - float(before[axis_name])) > 0.05:
                return False, "EXTERNAL_AXIS_CHANGE"

        if post_ready.get("motion_ready") is not True:
            return False, "POST_STATE_NOT_READY"
        if post_ready.get("coordinate_semantics_verified") is not True:
            return False, "COORDINATE_SEMANTICS_INVALID"
        if post_fw.get("absolute_mode") is not True:
            return False, "RELATIVE_MODE"
        if post_fw.get("firmware_session_id") != (ready.get("firmware") or {}).get("firmware_session_id"):
            return False, "FIRMWARE_SESSION_CHANGED"
        if abs(float((post_fw.get("workspace_offset") or {}).get(axis, 0.0))
               - float(((ready.get("firmware") or {}).get("workspace_offset") or {}).get(axis, 0.0))) > 1e-6:
            return False, "WORKSPACE_CHANGED"

        post_session = post_status.get("session") or {}
        if post_session.get("active") is not False:
            return False, "SESSION_STILL_ACTIVE"
        if post_session.get("state") not in {"RELEASED", "COMPLETED", "IDLE"}:
            return False, "SESSION_STATE_UNEXPECTED"
        post_hold = post_status.get("hold") or {}
        if post_hold.get("active") is not True:
            return False, "HOLD_INACTIVE"
        if self._hold_id and post_hold.get("hold_id") != self._hold_id:
            return False, "HOLD_OWNER_CHANGED"
        if post_hold.get("remaining_ms", 0) <= 0:
            return False, "HOLD_EXPIRED"

        if not post_ready.get("snapshot_id"):
            return False, "SNAPSHOT_MISSING"
        self._armed_snapshot_id = post_ready["snapshot_id"]
        return True, None

    def history(self, limit: int = 30) -> list[dict]:
        return self.audit.recent(limit)

    # ---------------------------------------------------------------- helpers
    def _reject(self, request_id, started, code, axis, direction, step, *,
                before=None, firmware=None) -> dict:
        return self._record(request_id, started, "REJECTED", code, axis, direction, step,
                            before, None, None, firmware=firmware)

    def _record(self, request_id, started, status, error, axis, direction, step,
                before, target, command, *, after=None, target_logical=None, delta=None,
                session=None, ground_after=None, logical_before=None, firmware=None,
                receipt_semantics=None, rebaselined=None, post_snapshot_id=None,
                rebaseline_reason=None, session_released=None, post_session_state=None) -> dict:
        record = {
            "request_id": request_id, "started_utc": started, "outcome": status,
            "error": error, "axis": axis, "direction": direction, "step_mm": step,
            "source": "manual_web", "protocol": "firmware-session-v1",
            "command": command, "before": before, "target": target, "after": after,
            "target_logical": target_logical, "delta": delta,
            "ground_after": ground_after, "logical_before": logical_before,
            "session": session, "firmware": firmware,
            "receipt_semantics": receipt_semantics or "serial_write_is_not_acceptance",
            "rebaselined": rebaselined,
            "post_snapshot_id": post_snapshot_id,
            "rebaseline_reason": rebaseline_reason,
            "session_released": session_released,
            "post_session_state": post_session_state,
        }
        self.audit.append(record)
        return {"ok": status in {"SUCCESS", "COMPLETED"},
                "status": status, "error": error, "request_id": request_id,
                "source": "manual_web", "protocol": "firmware-session-v1",
                "command": command, "before": before, "target": target, "after": after,
                "target_logical": target_logical, "delta": delta,
                "ground_after": ground_after, "session": session,
                "rebaselined": rebaselined,
                "post_snapshot_id": post_snapshot_id,
                "rebaseline_reason": rebaseline_reason,
                "session_released": session_released,
                "post_session_state": post_session_state,
                "receipt_semantics": receipt_semantics or "serial_write_is_not_acceptance",
                "message": ("已完成并复核" if status == "COMPLETED" else
                            "执行结果不确定，请以设备状态为准" if status == "UNKNOWN" else
                            "已拒绝/失败，未确认成功")}
