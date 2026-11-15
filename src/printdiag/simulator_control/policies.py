"""Safety policies for simulator jog control. No arbitrary G-code is accepted."""

from dataclasses import dataclass
from math import isfinite

from printdiag.simulator_control.config import BOUNDS_MM, STEPS_MM

AXES = ("x", "y", "z")


@dataclass(frozen=True)
class JogValidation:
    ok: bool
    error: str | None
    target: dict | None = None
    position: dict | None = None


def parse_position(state: dict) -> dict | None:
    """Extract finite simulator ground-truth position from GET /state."""
    try:
        kinematic = state["components"]["Cartesian Kinematic System"]["state"]
        position = kinematic["position"]
        values = {axis: float(position[axis]) for axis in AXES}
    except (KeyError, TypeError, ValueError):
        return None
    if not all(isfinite(value) for value in values.values()):
        return None
    return values


def idle_state(idle: dict) -> bool:
    return idle.get("idle") is True and idle.get("moves_queued", 0) == 0


def validate_jog(*, enabled: bool, armed: bool, state: dict | None, idle: dict | None,
                 axis: str, direction: str, step: float) -> JogValidation:
    if not enabled:
        return JogValidation(False, "SIM_CONTROL_DISABLED")
    if not armed:
        return JogValidation(False, "CONTROL_NOT_ARMED")
    if axis not in AXES:
        return JogValidation(False, "UNKNOWN_AXIS")
    if direction not in ("+", "-"):
        return JogValidation(False, "UNKNOWN_DIRECTION")
    if step not in STEPS_MM:
        return JogValidation(False, "INVALID_STEP")
    if state is None:
        return JogValidation(False, "DEVICE_STATE_UNAVAILABLE")
    position = parse_position(state)
    if position is None:
        return JogValidation(False, "POSITION_UNTRUSTED")
    if idle is None or idle.get("idle") is not True:
        return JogValidation(False, "DEVICE_BUSY")
    if idle.get("moves_queued", 0) != 0:
        return JogValidation(False, "DEVICE_BUSY")
    low, high = BOUNDS_MM[axis]
    target = dict(position)
    target[axis] = position[axis] + (step if direction == "+" else -step)
    if not (low <= target[axis] <= high):
        return JogValidation(False, "TARGET_OUT_OF_BOUNDS", target, position)
    if not (low <= position[axis] <= high):
        return JogValidation(False, "POSITION_OUT_OF_BOUNDS", target, position)
    return JogValidation(True, None, target, position)


def build_jog_command(axis: str, target: dict, feedrate: float) -> str:
    """One absolute G1 line; only the changed axis component is emitted."""
    value = target[axis]
    return f"G1 {axis.upper()}{value:.3f} F{feedrate:.0f}"
