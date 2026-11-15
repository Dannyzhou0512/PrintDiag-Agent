"""GPU model engine for the cloud inference service.

The engine is intentionally transport-agnostic: FastAPI calls it from a worker
thread. Heavy imports (torch/transformers/peft) happen only inside ``_load`` so
the service module can be imported and unit-tested on machines without GPUs.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import threading
import time
from typing import Any, Literal

Variant = Literal["base", "lora"]


class EngineUnavailable(RuntimeError):
    """A model variant cannot be served; the reason is exposed to the client."""

    def __init__(self, message: str, code: str = "MODEL_UNAVAILABLE") -> None:
        self.code = code
        super().__init__(message)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@dataclass(frozen=True)
class EngineSettings:
    base_path: str
    adapter_path: str | None
    device: str = "cuda"
    quantize_4bit: bool = False
    torch_dtype: str = "bfloat16"

    @classmethod
    def from_environment(cls, environ: dict[str, str]) -> "EngineSettings":
        base = environ.get("PRINTDIAG_CLOUD_BASE", "/data/models/Qwen3.5-4B")
        adapter = environ.get("PRINTDIAG_CLOUD_ADAPTER", "/data/outputs/qwen35_4b_lora_v03")
        return cls(
            base_path=base,
            adapter_path=adapter or None,
            device=environ.get("PRINTDIAG_CLOUD_DEVICE", "cuda"),
            quantize_4bit=environ.get("PRINTDIAG_CLOUD_QUANTIZE_4BIT") == "1",
            torch_dtype=environ.get("PRINTDIAG_CLOUD_DTYPE", "bfloat16"),
        )


class HfEngine:
    """Lazily loads and serves ``base`` and ``base+lora`` variants."""

    def __init__(self, settings: EngineSettings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._loaded: dict[str, dict[str, Any]] = {}
        self._adapter_sha = _sha256(Path(settings.adapter_path) / "adapter_model.safetensors") \
            if settings.adapter_path else None

    # ---- metadata -------------------------------------------------------
    def variants(self) -> dict[str, dict[str, Any]]:
        adapter_ok = bool(self.settings.adapter_path) and Path(
            self.settings.adapter_path, "adapter_config.json").is_file()
        return {
            "base": {
                "configured": Path(self.settings.base_path, "config.json").is_file(),
                "loaded": "base" in self._loaded,
                "adapter_sha256": None,
            },
            "lora": {
                "configured": adapter_ok and Path(self.settings.base_path, "config.json").is_file(),
                "loaded": "lora" in self._loaded,
                "adapter_sha256": self._adapter_sha,
            },
        }

    def gpu_status(self) -> dict[str, Any]:
        try:
            import torch  # noqa: PLC0415

            if not torch.cuda.is_available():
                return {"cuda": False}
            free, total = torch.cuda.mem_get_info()
            return {"cuda": True, "gpu": torch.cuda.get_device_name(0),
                    "free_mb": round(free / 1024 ** 2, 1), "total_mb": round(total / 1024 ** 2, 1)}
        except Exception:  # noqa: BLE001 - status must never crash the service
            return {"cuda": False, "reason": "torch unavailable"}

    def warmup(self) -> None:
        """Eagerly load both variants once at service startup (spec E1)."""
        with self._lock:
            self._load("base")
            self._load("lora")

    def model_metadata(self) -> dict[str, Any]:
        base = Path(self.settings.base_path)
        config = _read_json(base / "config.json") or {}
        architectures = config.get("architectures") or []
        return {
            "model_name": base.name,
            "architecture": architectures[0] if architectures else config.get("model_type"),
            "inference_dtype": "4bit-nf4" if self.settings.quantize_4bit else self.settings.torch_dtype,
            "adapter_sha256": self._adapter_sha,
        }

    # ---- loading --------------------------------------------------------
    def _load(self, variant: str) -> dict[str, Any]:
        if variant in self._loaded:
            return self._loaded[variant]
        base = Path(self.settings.base_path)
        if not (base / "config.json").is_file():
            raise EngineUnavailable(f"base model is not available at {base}")
        if variant == "lora":
            adapter = Path(self.settings.adapter_path or "")
            if not (adapter / "adapter_config.json").is_file() or not (
                    adapter / "adapter_model.safetensors").is_file():
                raise EngineUnavailable(f"LoRA adapter is not available at {adapter}")
        started = time.perf_counter()
        import torch  # noqa: PLC0415
        from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: PLC0415

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}.get(self.settings.torch_dtype, torch.bfloat16)
        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self.settings.device == "cuda" and torch.cuda.is_available():
            kwargs.update(device_map="auto", torch_dtype=dtype)
        else:
            kwargs.update(torch_dtype=torch.float32)
        if self.settings.quantize_4bit:
            from transformers import BitsAndBytesConfig  # noqa: PLC0415
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        model = AutoModelForImageTextToText.from_pretrained(str(base), **kwargs)
        if variant == "lora":
            from peft import PeftModel  # noqa: PLC0415
            model = PeftModel.from_pretrained(model, self.settings.adapter_path, is_trainable=False)
        model.eval()
        processor = AutoProcessor.from_pretrained(str(base), trust_remote_code=True)
        gpu_peak = None
        if torch.cuda.is_available():
            gpu_peak = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
        info = {
            "model": model, "processor": processor,
            "load_ms": round((time.perf_counter() - started) * 1000, 2),
            "gpu_peak_mb": gpu_peak, "variant": variant,
            "adapter_sha256": self._adapter_sha if variant == "lora" else None,
        }
        self._loaded[variant] = info
        return info

    def unload(self, variant: str | None = None) -> None:
        with self._lock:
            targets = [variant] if variant else list(self._loaded)
            for name in targets:
                self._loaded.pop(name, None)
            try:
                import torch  # noqa: PLC0415
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    # ---- generation -----------------------------------------------------
    def generate(self, variant: str, messages: list[dict[str, str]],
                 options: dict[str, Any]) -> dict[str, Any]:
        if variant not in ("base", "lora"):
            raise EngineUnavailable(f"unknown variant: {variant}", code="INVALID_ARGUMENT")
        with self._lock:
            info = self._load(variant)
            model = info["model"]
            processor = info["processor"]
            started = time.perf_counter()
            import torch  # noqa: PLC0415

            try:
                prompt = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=bool(options.get("enable_thinking", False)))
            except TypeError:
                prompt = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            encoded = processor(text=[prompt], return_tensors="pt")
            if hasattr(encoded, "to"):
                encoded = encoded.to(model.device)
            seed = int(options.get("seed", 42))
            torch.manual_seed(seed)
            do_sample = bool(options.get("do_sample", False))
            with torch.no_grad():
                output = model.generate(
                    **encoded,
                    max_new_tokens=int(options.get("max_new_tokens", 512)),
                    do_sample=do_sample,
                    temperature=float(options.get("temperature", 0.0)) if do_sample else None,
                    top_p=float(options.get("top_p", 1.0)) if do_sample else None,
                    pad_token_id=getattr(processor, "pad_token_id", None) or 0,
                )
            new_tokens = output[0][encoded["input_ids"].shape[-1]:]
            text = processor.decode(new_tokens, skip_special_tokens=True)
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            gpu_peak = None
            if self.settings.device == "cuda" and torch.cuda.is_available():
                gpu_peak = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
            return {
                "text": text,
                "variant": variant,
                "model_id": f"{self.settings.base_path}#{variant}",
                "adapter_sha256": info["adapter_sha256"],
                "latency_ms": latency_ms,
                "gpu_peak_mb": gpu_peak,
                "load_ms": info["load_ms"],
                "input_tokens": int(encoded["input_ids"].shape[-1]),
                "output_tokens": int(new_tokens.shape[-1]),
                "generated_at_utc": datetime.now(UTC).isoformat(),
            }


class FakeEngine:
    """Deterministic engine for offline tests; never used in deployment."""

    def __init__(self, text: str = '{"fault_category":"喷嘴堵塞"}') -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []
        self.unloaded = False
        self._warmed = False

    def variants(self) -> dict[str, dict[str, Any]]:
        return {"base": {"configured": True, "loaded": self._warmed, "adapter_sha256": None},
                "lora": {"configured": True, "loaded": self._warmed,
                         "adapter_sha256": "fake-sha"}}

    def gpu_status(self) -> dict[str, Any]:
        return {"cuda": False, "reason": "fake engine"}

    def generate(self, variant: str, messages: list[dict[str, str]],
                 options: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"variant": variant, "messages": messages, "options": options})
        return {"text": self.text, "variant": variant, "model_id": f"fake#{variant}",
                "adapter_sha256": "fake-sha" if variant == "lora" else None,
                "latency_ms": 1.0, "gpu_peak_mb": None, "load_ms": 0.1,
                "input_tokens": len(json.dumps(messages)), "output_tokens": len(self.text),
                "generated_at_utc": datetime.now(UTC).isoformat()}

    def unload(self, variant: str | None = None) -> None:
        self.unloaded = True

    def warmup(self) -> None:
        self._warmed = True

    def model_metadata(self) -> dict[str, Any]:
        return {"model_name": "fake", "architecture": "FakeForTesting",
                "inference_dtype": "none", "adapter_sha256": "fake-sha"}
