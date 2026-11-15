"""Pluggable inference backends for the local Qwen3.5 LoRA path.

``TransformersBackend`` is the real backend. It validates model/adapter/
architecture/dependencies before importing anything heavy and raises
``BackendUnavailable`` with precise blockers instead of pretending to load.
``FakeBackend`` is test-only and can never report ``live_inference=True``.
"""

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from datetime import UTC, datetime
import importlib.util
import json
from pathlib import Path
import time
from typing import Any, Literal, Protocol

BackendMode = Literal["base", "lora", "fake"]


class BackendUnavailable(RuntimeError):
    """The backend cannot load; ``blockers`` explains exactly what is missing."""

    def __init__(self, blockers: Sequence[str]) -> None:
        self.blockers = tuple(blockers)
        super().__init__("inference backend unavailable: " + "; ".join(self.blockers))


class GenerationTimeout(RuntimeError):
    """A generation call exceeded its configured wall-clock budget."""


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    seed: int = 42
    timeout_s: float = 120.0
    enable_thinking: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.max_new_tokens <= 4096:
            raise ValueError("max_new_tokens must be in 1..4096")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be in 0..2")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0,1]")
        if not 0.0 < self.timeout_s <= 3600.0:
            raise ValueError("timeout_s must be positive")

    def as_dict(self) -> dict:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            "seed": self.seed,
            "timeout_s": self.timeout_s,
            "enable_thinking": self.enable_thinking,
        }


@dataclass(frozen=True)
class ModelInfo:
    backend: str
    mode: BackendMode
    model_path: str | None
    adapter_path: str | None
    adapter_sha256: str | None
    dtype: str | None
    device: str | None
    load_ms: float
    live_inference_capable: bool
    architecture: str | None = None
    gpu_peak_mb: float | None = None

    def as_dict(self) -> dict:
        return {
            "backend": self.backend, "mode": self.mode, "model_path": self.model_path,
            "adapter_path": self.adapter_path, "adapter_sha256": self.adapter_sha256,
            "dtype": self.dtype, "device": self.device, "load_ms": self.load_ms,
            "live_inference_capable": self.live_inference_capable,
            "architecture": self.architecture, "gpu_peak_mb": self.gpu_peak_mb,
        }


@dataclass(frozen=True)
class GenerationResult:
    text: str
    backend: str
    mode: BackendMode
    live_inference: bool
    latency_ms: float
    gpu_peak_mb: float | None
    prompt_chars: int
    generated_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    input_tokens: int | None = None
    output_tokens: int | None = None
    request_id: str | None = None
    model_metadata: dict | None = None

    def as_dict(self) -> dict:
        return {
            "backend": self.backend, "mode": self.mode, "live_inference": self.live_inference,
            "latency_ms": self.latency_ms, "gpu_peak_mb": self.gpu_peak_mb,
            "prompt_chars": self.prompt_chars, "generated_at_utc": self.generated_at_utc,
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "request_id": self.request_id, "model_metadata": self.model_metadata,
        }


class InferenceBackend(Protocol):
    name: str
    mode: BackendMode
    live_inference_capable: bool

    def load(self) -> ModelInfo: ...
    def unload(self) -> None: ...
    def is_loaded(self) -> bool: ...
    def get_model_info(self) -> ModelInfo | None: ...
    def generate(self, messages: Sequence[dict[str, str]], config: GenerationConfig) -> GenerationResult: ...


def _module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _sha256(path: Path) -> str | None:
    from hashlib import sha256
    if not path.is_file():
        return None
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class TransformersBackend:
    """Real PyTorch/Transformers/PEFT backend with pre-flight validation."""

    name = "transformers"
    live_inference_capable = True

    def __init__(self, model_path: str | Path | None, adapter_path: str | Path | None = None,
                 *, quantize_4bit: bool = False, device: str = "cuda") -> None:
        self._model_path = Path(model_path) if model_path else None
        self._adapter_path = Path(adapter_path) if adapter_path else None
        self._quantize = quantize_4bit
        self._device = device
        self.mode: BackendMode = "lora" if self._adapter_path else "base"
        self._model: Any = None
        self._tokenizer: Any = None
        self._info: ModelInfo | None = None
        self._blockers: tuple[str, ...] = ()

    # ---- pre-flight -----------------------------------------------------
    def preflight(self) -> tuple[str, ...]:
        """Validation that needs no heavy imports; returns precise blockers."""
        blockers: list[str] = []
        if self._model_path is None:
            blockers.append("base model path is not configured (PRINTDIAG_BASE_MODEL_PATH)")
        elif not self._model_path.is_dir():
            blockers.append(f"base model weights not available locally: {self._model_path}")
        else:
            config = _read_json(self._model_path / "config.json")
            if config is None:
                blockers.append(f"base model config.json missing or invalid: {self._model_path}")
            else:
                architectures = [str(item) for item in config.get("architectures") or []]
                model_type = str(config.get("model_type") or "")
                if not any("Qwen3_5" in arch for arch in architectures) and not model_type.startswith("qwen3_5"):
                    blockers.append(
                        "base model architecture is not Qwen3_5; refusing a name-similar but "
                        f"incompatible base (model_type={model_type!r}, architectures={architectures})")
        if self._adapter_path is not None:
            if not (self._adapter_path / "adapter_config.json").is_file():
                blockers.append(f"adapter_config.json missing: {self._adapter_path}")
            if not (self._adapter_path / "adapter_model.safetensors").is_file():
                blockers.append(f"adapter_model.safetensors missing: {self._adapter_path}")
            adapter_config = _read_json(self._adapter_path / "adapter_config.json") or {}
            targets = [str(item) for item in adapter_config.get("target_modules") or []]
            if self.mode == "lora" and targets and not any(
                    target.startswith("model.language_model.") for target in targets):
                blockers.append("adapter target_modules do not match a Qwen3_5 multimodal base")
            if adapter_config.get("peft_type") not in (None, "LORA"):
                blockers.append(f"unsupported adapter peft_type: {adapter_config.get('peft_type')}")
        missing = [name for name in ("torch", "transformers", "peft") if not _module(name)]
        if self._quantize and not _module("bitsandbytes"):
            missing.append("bitsandbytes")
        if missing:
            blockers.append("missing inference dependencies: " + ", ".join(sorted(set(missing))))
        return tuple(blockers)

    # ---- lifecycle ------------------------------------------------------
    def load(self) -> ModelInfo:
        blockers = self.preflight()
        self._blockers = blockers
        if blockers:
            raise BackendUnavailable(blockers)
        started = time.perf_counter()
        import torch  # noqa: PLC0415 - only imported when dependencies exist
        from peft import PeftModel  # noqa: PLC0415
        from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: PLC0415

        kwargs: dict[str, Any] = {"trust_remote_code": True}
        dtype = "auto"
        if self._device == "cuda" and torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.bfloat16
            dtype = "bfloat16"
            kwargs["device_map"] = "auto"
        else:
            self._device = "cpu"
            dtype = "float32"
        if self._quantize:
            from transformers import BitsAndBytesConfig  # noqa: PLC0415
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
            dtype = "4bit-nf4"
        model = AutoModelForImageTextToText.from_pretrained(str(self._model_path), **kwargs)
        if self._adapter_path is not None:
            model = PeftModel.from_pretrained(model, str(self._adapter_path), is_trainable=False)
        model.eval()
        self._tokenizer = AutoProcessor.from_pretrained(str(self._model_path), trust_remote_code=True)
        self._model = model
        load_ms = (time.perf_counter() - started) * 1000
        gpu_peak = None
        if self._device == "cuda":
            gpu_peak = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
        config = _read_json((self._model_path or Path()) / "config.json") or {}
        architecture = (config.get("architectures") or [None])[0]
        self._info = ModelInfo(
            backend=self.name, mode=self.mode, model_path=str(self._model_path),
            adapter_path=str(self._adapter_path) if self._adapter_path else None,
            adapter_sha256=_sha256((self._adapter_path or Path()) / "adapter_model.safetensors")
            if self._adapter_path else None,
            dtype=dtype, device=self._device, load_ms=round(load_ms, 2),
            live_inference_capable=True, architecture=architecture, gpu_peak_mb=gpu_peak)
        return self._info

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._info = None
        try:  # best-effort cleanup without hard dependency on torch
            import torch  # noqa: PLC0415
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - unload must never raise
            pass

    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def get_model_info(self) -> ModelInfo | None:
        return self._info

    # ---- generation -----------------------------------------------------
    def generate(self, messages: Sequence[dict[str, str]],
                 config: GenerationConfig) -> GenerationResult:
        if not self.is_loaded():
            raise BackendUnavailable(self._blockers or ("backend is not loaded",))
        import torch  # noqa: PLC0415

        def _do_generate() -> GenerationResult:
            started = time.perf_counter()
            chat = list(messages)
            try:
                prompt = self._tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True,
                    enable_thinking=config.enable_thinking)
            except TypeError:
                prompt = self._tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True)
            encoded = self._tokenizer(text=[prompt], return_tensors="pt")
            if hasattr(encoded, "to"):
                encoded = encoded.to(self._model.device)
            torch.manual_seed(config.seed)
            with torch.no_grad():
                output = self._model.generate(
                    **encoded, max_new_tokens=config.max_new_tokens,
                    do_sample=config.do_sample,
                    temperature=config.temperature if config.do_sample else None,
                    top_p=config.top_p if config.do_sample else None,
                    pad_token_id=getattr(self._tokenizer, "pad_token_id", None) or 0,
                )
            new_tokens = output[0][encoded["input_ids"].shape[-1]:]
            text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
            latency_ms = (time.perf_counter() - started) * 1000
            gpu_peak = None
            if self._device == "cuda":
                gpu_peak = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
            return GenerationResult(
                text=text, backend=self.name, mode=self.mode, live_inference=True,
                latency_ms=round(latency_ms, 2), gpu_peak_mb=gpu_peak,
                prompt_chars=len(prompt), input_tokens=int(encoded["input_ids"].shape[-1]),
                output_tokens=int(new_tokens.shape[-1]))

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_do_generate)
            try:
                return future.result(timeout=config.timeout_s)
            except FutureTimeout:
                raise GenerationTimeout(
                    f"generation exceeded {config.timeout_s}s; the CUDA work may still be winding down"
                ) from None


class FakeBackend:
    """Test-only backend. Never reports live inference and records its calls."""

    name = "fake"
    mode: BackendMode = "fake"
    live_inference_capable = False

    def __init__(self, outputs: str | Sequence[str] | Callable[[Sequence[dict[str, str]]], str],
                 *, load_error: Exception | None = None, sleep_s: float = 0.0) -> None:
        self._outputs = [outputs] if isinstance(outputs, str) else (
            outputs if not callable(outputs) else [outputs])
        self._callable = outputs if callable(outputs) else None
        self._load_error = load_error
        self._sleep_s = sleep_s
        self._loaded = False
        self._info: ModelInfo | None = None
        self.calls: list[Sequence[dict[str, str]]] = []
        self.unloaded = False

    def load(self) -> ModelInfo:
        if self._load_error is not None:
            raise self._load_error
        self._loaded = True
        self._info = ModelInfo(backend=self.name, mode="fake", model_path=None, adapter_path=None,
                               adapter_sha256=None, dtype=None, device="cpu", load_ms=0.01,
                               live_inference_capable=False)
        return self._info

    def unload(self) -> None:
        self._loaded = False
        self._info = None
        self.unloaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def get_model_info(self) -> ModelInfo | None:
        return self._info

    def generate(self, messages: Sequence[dict[str, str]],
                 config: GenerationConfig) -> GenerationResult:
        if not self._loaded:
            raise BackendUnavailable(("fake backend is not loaded",))
        if self._sleep_s:
            time.sleep(self._sleep_s)
        self.calls.append(messages)
        started = time.perf_counter()
        if self._callable is not None:
            text = self._callable(messages)
        else:
            text = self._outputs.pop(0) if len(self._outputs) > 1 else self._outputs[0]
        latency_ms = (time.perf_counter() - started) * 1000
        return GenerationResult(text=text, backend=self.name, mode="fake", live_inference=False,
                               latency_ms=round(latency_ms, 2), gpu_peak_mb=None,
                               prompt_chars=sum(len(item.get("content", "")) for item in messages))
