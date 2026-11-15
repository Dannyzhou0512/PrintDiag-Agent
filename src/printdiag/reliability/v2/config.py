"""V2 configuration; disabled by default so V0.3 remains reproducible."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class V2Config:
    enabled: bool = False
    allow_hypothesis_without_tool: bool = True
    downgrade_contradicted: bool = True
    extended_taxonomy: bool = False
    version: str = "v2.0-offline"

    @classmethod
    def from_environment(cls, environ: dict[str, str] | None = None) -> "V2Config":
        env = environ if environ is not None else dict(os.environ)
        return cls(
            enabled=env.get("PRINTDIAG_RELIABILITY_V2") == "1",
            allow_hypothesis_without_tool=env.get(
                "PRINTDIAG_V2_ALLOW_HYPOTHESIS_WITHOUT_TOOL", "1") == "1",
            downgrade_contradicted=env.get("PRINTDIAG_V2_DOWNGRADE_CONTRADICTED", "1") == "1",
            extended_taxonomy=env.get("PRINTDIAG_V2_EXTENDED_TAXONOMY") == "1",
        )

    def as_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "allow_hypothesis_without_tool": self.allow_hypothesis_without_tool,
            "downgrade_contradicted": self.downgrade_contradicted,
            "extended_taxonomy": self.extended_taxonomy,
            "version": self.version,
        }
