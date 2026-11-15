"""Qwen3.5 LoRA artifacts, runtime blockers and archived-prediction replay.

Real local inference is only attempted when the adapter, a compatible base model
and the inference dependencies are present. This machine currently has the
adapter but not the base weights (`/data/models/Qwen3.5-4B` is a remote training
path) nor torch/transformers/peft, so the provider fails explicitly instead of
pretending to run. Archived predictions can still be replayed and are labelled
``archived replay`` everywhere.
"""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
from typing import Literal

from printdiag.agents.diagnosis_prompt import (
    TrainingTemplate, build_diagnosis_messages, load_training_template,
)
from printdiag.agents.inference_backend import (
    BackendUnavailable, GenerationConfig, GenerationResult, GenerationTimeout, InferenceBackend,
    ModelInfo, TransformersBackend,
)
from printdiag.agents.providers import SFTDraftTurn
from printdiag.agents.symptom_routing import RuleBasedProvider, knowledge_hits, observation_summary
from printdiag.diagnostics.fault_categories import CategoryMatch, normalize_fault_category
from printdiag.schemas.agent import ToolResult
from printdiag.schemas.sft_diagnosis import (
    DiagnosisValidationError, SFTDiagnosisDraft, validate_diagnosis,
)

ROOT = Path(__file__).resolve().parents[3]


def _project_root_with(relative: str) -> Path:
    """Artifacts/datasets live in the main checkout, not in this worktree."""
    for candidate in (ROOT, *ROOT.parents[:3]):
        if (candidate / relative).exists():
            return candidate
    return ROOT


_ARTIFACT_BASE = _project_root_with("artifacts/qwen35_lora_v03")
_DATASET_BASE = _project_root_with("dataset_SFT")
ARTIFACT_ROOT = _ARTIFACT_BASE / "artifacts" / "qwen35_lora_v03"
ADAPTER_DIR = ARTIFACT_ROOT / "qwen35_4b_lora_v03"
PREDICTIONS = ARTIFACT_ROOT / "printdiag_test_30_predictions.jsonl"
TRAIN_CONFIG = ARTIFACT_ROOT / "configs" / "printdiag_qwen35_v03.yaml"
TRAIN_LOG = ARTIFACT_ROOT / "qwen35_v03_train.log"
DATASET_TEST = (_DATASET_BASE / "dataset_SFT" / "PrintDiag_Qwen3_4B_SFT_V02"
                / "PrintDiag_SFT_V02_Ready" / "printdiag_qwen3_4b_sft_test.jsonl")

Status = Literal["ready", "archived_replay_only"]


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@dataclass(frozen=True)
class ArtifactReport:
    artifact_root: Path
    adapter_dir: Path
    adapter_config: Path
    adapter_weights: Path
    adapter_sha256: str | None
    base_model_name_or_path: str | None
    peft_type: str | None
    rank: int | None
    lora_alpha: int | None
    peft_version: str | None
    target_module_count: int
    train_config: Path
    train_log: Path
    predictions: Path
    dataset_test: Path
    dependencies: dict[str, bool]
    base_model_present: bool
    blockers: tuple[str, ...]
    status: Status

    @property
    def adapter_ready(self) -> bool:
        return self.adapter_config.is_file() and self.adapter_weights.is_file()

    def as_dict(self) -> dict:
        return {
            "artifact_root": str(self.artifact_root),
            "adapter_dir": str(self.adapter_dir),
            "adapter_ready": self.adapter_ready,
            "adapter_sha256": self.adapter_sha256,
            "base_model_name_or_path": self.base_model_name_or_path,
            "peft_type": self.peft_type,
            "rank": self.rank,
            "lora_alpha": self.lora_alpha,
            "peft_version": self.peft_version,
            "target_module_count": self.target_module_count,
            "train_config": str(self.train_config) if self.train_config.is_file() else None,
            "train_log": str(self.train_log) if self.train_log.is_file() else None,
            "predictions": str(self.predictions) if self.predictions.is_file() else None,
            "dataset_test": str(self.dataset_test) if self.dataset_test.is_file() else None,
            "dependencies": self.dependencies,
            "base_model_present": self.base_model_present,
            "blockers": list(self.blockers),
            "status": self.status,
        }


def inspect_artifacts(artifact_root: Path = ARTIFACT_ROOT) -> ArtifactReport:
    adapter_dir = artifact_root / "qwen35_4b_lora_v03"
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    base_name = None
    peft_type = None
    rank = None
    alpha = None
    peft_version = None
    target_count = 0
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            base_name = config.get("base_model_name_or_path")
            peft_type = config.get("peft_type")
            rank = config.get("r")
            alpha = config.get("lora_alpha")
            peft_version = config.get("peft_version")
            target_count = len(config.get("target_modules") or ())
        except ValueError:
            pass
    base_present = False
    if base_name:
        candidate = Path(base_name)
        base_present = candidate.is_dir() and (candidate / "config.json").is_file()
    dependencies = {name: _module(name) for name in
                    ("torch", "transformers", "peft", "accelerate", "bitsandbytes")}
    blockers = []
    if not config_path.is_file() or not weights_path.is_file():
        blockers.append("adapter weights or adapter_config.json missing")
    if not base_present:
        blockers.append(f"base model weights not available locally: {base_name}")
    missing = [name for name, present in dependencies.items() if not present]
    if missing:
        blockers.append("missing inference dependencies: " + ", ".join(missing))
    status: Status = "ready" if not blockers else "archived_replay_only"
    return ArtifactReport(
        artifact_root=artifact_root, adapter_dir=adapter_dir, adapter_config=config_path,
        adapter_weights=weights_path, adapter_sha256=_sha256(weights_path),
        base_model_name_or_path=base_name, peft_type=peft_type, rank=rank, lora_alpha=alpha,
        peft_version=peft_version, target_module_count=target_count,
        train_config=artifact_root / "configs" / "printdiag_qwen35_v03.yaml",
        train_log=artifact_root / "qwen35_v03_train.log",
        predictions=artifact_root / "printdiag_test_30_predictions.jsonl",
        dataset_test=DATASET_TEST, dependencies=dependencies, base_model_present=base_present,
        blockers=tuple(blockers), status=status,
    )


class LocalModelUnavailable(RuntimeError):
    """Local inference cannot run on this machine; the reasons are explicit."""

    def __init__(self, blockers: tuple[str, ...]) -> None:
        self.blockers = blockers
        super().__init__("local LoRA inference unavailable: " + "; ".join(blockers))


class LocalLoRAProvider:
    """Local Qwen3.5 Base / Base+LoRA provider.

    Without a backend it stays in the P10-3 state (artifact audit only, explicit
    ``LocalModelUnavailable``). With a backend it loads the model, builds the
    training-format prompt from this diagnosis's observations and RAG hits, and
    calls ``generate()``. ``live_inference`` is only true when the backend
    actually generated (FakeBackend can never set it).
    """

    mock = False

    def __init__(self, backend: InferenceBackend | None = None, *,
                 report: ArtifactReport | None = None,
                 generation: GenerationConfig | None = None,
                 template: TrainingTemplate | None = None,
                 include_rag: bool = True) -> None:
        self._backend = backend
        self.report = report or inspect_artifacts()
        self._generation = generation or GenerationConfig()
        self._template = template
        self._include_rag = include_rag
        self._model_info: ModelInfo | None = None

    @property
    def status(self) -> str:
        if self._backend is None:
            return self.report.status
        return "ready" if self._backend.is_loaded() else "not_loaded"

    @property
    def backend_name(self) -> str | None:
        return None if self._backend is None else self._backend.name

    @property
    def mode(self) -> str | None:
        return None if self._backend is None else self._backend.mode

    @property
    def live_inference_capable(self) -> bool:
        return bool(self._backend and self._backend.live_inference_capable)

    def prepare(self) -> ModelInfo:
        if self._backend is None:
            raise LocalModelUnavailable(self.report.blockers)
        if self._backend.is_loaded():
            assert self._model_info is not None
            return self._model_info
        self._model_info = self._backend.load()
        return self._model_info

    def unload(self) -> None:
        if self._backend is not None:
            self._backend.unload()
        self._model_info = None

    def generate_draft(self, symptom: str, results: Sequence[ToolResult], *,
                       include_rag: bool | None = None) -> "DraftGeneration":
        if self._backend is None:
            raise LocalModelUnavailable(self.report.blockers)
        if not self._backend.is_loaded():
            raise LocalModelUnavailable(("inference backend is not loaded; call prepare() first",))
        template = self._template
        if template is None:
            template = load_training_template()
            self._template = template
        rag = self._include_rag if include_rag is None else include_rag
        knowledge = [
            {"document_id": hit.document_id, "chunk_id": hit.chunk_id or hit.anchor,
             "excerpt": hit.excerpt, "source_reference": hit.source_reference}
            for hit in knowledge_hits(results)
        ]
        messages, audit = build_diagnosis_messages(
            symptom, observation_summary(results), knowledge,
            include_rag=rag, template=template)
        result: GenerationResult = self._generate_with_timeout(messages)
        return DraftGeneration(
            raw_output=result.text,
            messages=tuple(messages),
            backend=result.backend,
            mode=result.mode,
            live_inference=bool(result.live_inference and self._backend.live_inference_capable),
            latency_ms=result.latency_ms,
            gpu_peak_mb=result.gpu_peak_mb,
            prompt_chars=result.prompt_chars,
            prompt_audit=audit,
            model_info=self._model_info.as_dict() if self._model_info else None,
            generated_at_utc=result.generated_at_utc,
            request_id=result.request_id,
            model_metadata=result.model_metadata,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )

    async def next_turn(self, request, tools, history):
        if self._backend is None:
            raise LocalModelUnavailable(self.report.blockers)
        results = [result for exchange in history for result in exchange.results]
        generation = self.generate_draft(request.symptom, results)
        return SFTDraftTurn(generation.raw_output)

    def _generate_with_timeout(self, messages: Sequence[dict[str, str]]) -> GenerationResult:
        """Outer wall-clock guard; the backend also enforces its own timeout."""
        if self._backend is None:
            raise LocalModelUnavailable(self.report.blockers)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._backend.generate, messages, self._generation)
            try:
                return future.result(timeout=self._generation.timeout_s)
            except FutureTimeout:
                raise GenerationTimeout(
                    f"generation exceeded {self._generation.timeout_s}s") from None

    @classmethod
    def from_environment(cls, *, report: ArtifactReport | None = None) -> "LocalLoRAProvider":
        base_model = os.environ.get("PRINTDIAG_BASE_MODEL_PATH", "").strip() or None
        adapter = os.environ.get("PRINTDIAG_LORA_ADAPTER_PATH", "").strip() or None
        if adapter is None and ADAPTER_DIR.is_dir():
            adapter = str(ADAPTER_DIR)
        quantize = os.environ.get("PRINTDIAG_QUANTIZE_4BIT") == "1"
        device = os.environ.get("PRINTDIAG_INFERENCE_DEVICE", "cuda")
        max_tokens = int(os.environ.get("PRINTDIAG_MAX_NEW_TOKENS", "512"))
        timeout = float(os.environ.get("PRINTDIAG_GENERATION_TIMEOUT_S", "120"))
        backend = TransformersBackend(base_model, adapter, quantize_4bit=quantize, device=device)
        return cls(backend, report=report, generation=GenerationConfig(
            max_new_tokens=max_tokens, timeout_s=timeout,
            enable_thinking=os.environ.get("PRINTDIAG_ENABLE_THINKING") == "1"))


@dataclass(frozen=True)
class DraftGeneration:
    """Raw model output plus auditable inference metadata (preserved on failure)."""

    raw_output: str
    messages: tuple[dict[str, str], ...]
    backend: str
    mode: str
    live_inference: bool
    latency_ms: float
    gpu_peak_mb: float | None
    prompt_chars: int
    prompt_audit: dict
    model_info: dict | None
    generated_at_utc: str
    request_id: str | None = None
    model_metadata: dict | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    def as_dict(self) -> dict:
        return {
            "raw_output": self.raw_output,
            "raw_output_chars": len(self.raw_output),
            "backend": self.backend,
            "mode": self.mode,
            "live_inference": self.live_inference,
            "latency_ms": self.latency_ms,
            "gpu_peak_mb": self.gpu_peak_mb,
            "prompt_chars": self.prompt_chars,
            "prompt_audit": self.prompt_audit,
            "model_info": self.model_info,
            "generated_at_utc": self.generated_at_utc,
            "request_id": self.request_id,
            "model_metadata": self.model_metadata,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


def make_lora_draft_provider(provider: LocalLoRAProvider, *, include_rag: bool = True,
                             label: str | None = None):
    """Rule-based tools + real local generation, wired into the existing workflow."""
    from printdiag.agents.composite import EvidenceGroundedDraftProvider  # local import
    from printdiag.agents.providers import ProviderFailure
    from printdiag.agents.remote_backend import RemoteBackendError

    async def drafter(request, history):
        results = [result for exchange in history for result in exchange.results]
        try:
            generation = provider.generate_draft(request.symptom, results, include_rag=include_rag)
        except (LocalModelUnavailable, BackendUnavailable, GenerationTimeout,
                RemoteBackendError) as exc:
            # Degrade honestly instead of crashing the HTTP handler.
            raise ProviderFailure("PROVIDER_UNAVAILABLE") from exc
        return generation.raw_output

    base_label = label or f"local {provider.backend_name or 'lora'} ({provider.mode or 'unknown'})"
    return EvidenceGroundedDraftProvider(RuleBasedProvider(), drafter, label=base_label)


# ----------------------------------------------------------------- archived replay

def load_archived_predictions(path: Path = PREDICTIONS, variant: str | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if variant is not None:
        rows = [row for row in rows if row.get("variant") == variant]
    return rows


@dataclass(frozen=True)
class DraftReplay:
    sample_id: int
    variant: str
    user: str
    answer: str
    valid: bool
    error_code: str | None
    error_field: str | None
    draft: SFTDiagnosisDraft | None
    category: CategoryMatch | None

    @property
    def category_canonical(self) -> str | None:
        return self.category.canonical if self.category else None


def replay_draft(row: dict) -> DraftReplay:
    """Validate one archived answer with the strict V0.3 contract (no repair)."""
    answer = row.get("answer", "")
    try:
        draft = validate_diagnosis(answer)
    except DiagnosisValidationError as exc:
        return DraftReplay(int(row["sample_id"]), str(row["variant"]), row.get("user", ""),
                           answer, False, exc.code, exc.field, None, None)
    return DraftReplay(int(row["sample_id"]), str(row["variant"]), row.get("user", ""),
                       answer, True, None, None, draft,
                       normalize_fault_category(draft.fault_category))
