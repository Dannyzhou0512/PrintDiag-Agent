"""Configuration for the simulator-only, human-operated control service.

Defaults to disabled. The target is fixed to the local Marlin Simulator; no
arbitrary URL/G-code can be supplied by the web client or the Agent.
"""

from dataclasses import dataclass, field
import os

SIMULATOR_URL = "http://127.0.0.1:8100"
BOUNDS_MM = {"x": (0.0, 400.0), "y": (0.0, 400.0), "z": (0.0, 500.0)}
STEPS_MM = (0.1, 1.0, 10.0)
FEEDRATE_MM_MIN = 600.0
IDLE_TIMEOUT_S = 8.0
POSITION_TOLERANCE_MM = 0.25


@dataclass(frozen=True)
class ControlConfig:
    enabled: bool
    simulator_url: str = SIMULATOR_URL
    idle_timeout_s: float = IDLE_TIMEOUT_S
    feedrate_mm_min: float = FEEDRATE_MM_MIN
    bounds: dict = field(default_factory=lambda: dict(BOUNDS_MM))

    @classmethod
    def from_environment(cls, environ: dict[str, str] | None = None) -> "ControlConfig":
        env = environ if environ is not None else dict(os.environ)
        return cls(enabled=env.get("PRINTDIAG_SIM_CONTROL_ENABLED") == "1",
                   simulator_url=SIMULATOR_URL, bounds=dict(BOUNDS_MM))

    def as_dict(self) -> dict:
        return {"enabled": self.enabled, "simulator_url": self.simulator_url,
                "bounds": self.bounds, "steps_mm": list(STEPS_MM),
                "feedrate_mm_min": self.feedrate_mm_min,
                "idle_timeout_s": self.idle_timeout_s,
                "home_supported": False, "stop_supported": False,
                "temperature_control": False, "agent_control_allowed": False}
