"""Strict request/response schemas for the cloud inference API."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

MAX_MESSAGES = 4
MAX_CONTENT_CHARS = 20000
MAX_NEW_TOKENS = 1024


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ChatMessage(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: Annotated[str, StringConstraints(min_length=1, max_length=MAX_CONTENT_CHARS)]


class InferenceOptions(StrictModel):
    """Whitelisted generation knobs; no arbitrary kwargs reach transformers."""

    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    enable_thinking: bool = False


class GenerateRequest(StrictModel):
    request_id: str | None = Field(default=None, max_length=80)
    mode: Literal["base", "lora"] = "lora"
    messages: list[ChatMessage] = Field(min_length=1, max_length=MAX_MESSAGES)
    max_new_tokens: int = Field(default=512, ge=1, le=MAX_NEW_TOKENS)
    do_sample: bool = False
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    inference_options: InferenceOptions = Field(default_factory=InferenceOptions)


class ModelMetadata(StrictModel):
    model_name: str
    architecture: str | None
    adapter_sha256: str | None
    inference_dtype: str
    mode: str


class GenerateResponse(StrictModel):
    request_id: str
    mode: str
    raw_output: str
    input_tokens: int | None
    output_tokens: int | None
    inference_latency_ms: float
    model_metadata: ModelMetadata
    live_inference: bool
    error: str | None = None


class HealthResponse(StrictModel):
    ok: bool = True
    service_status: str
    service: str
    service_version: str
    model_ready: bool
    adapter_ready: bool


class ModelInfoResponse(StrictModel):
    model_name: str
    architecture: str | None
    adapter_loaded: bool
    adapter_sha256: str | None
    available_modes: list[str]
    inference_dtype: str
    service_version: str
    service: str
