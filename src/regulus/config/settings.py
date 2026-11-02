"""Configuration loading.

Precedence (highest first):
    1. Environment variables (REGULUS_*), optionally loaded from a .env file
    2. YAML file (configs/model.yaml by default)
    3. Dataclass defaults

Secrets (e.g. HF_TOKEN) are never stored here; huggingface_hub reads them
from the environment itself.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "model.yaml"

ENV_PREFIX = "REGULUS_"
VALID_DTYPES = {"auto", "float32", "float16", "bfloat16"}
_DEVICE_RE = re.compile(r"^(auto|cpu|mps|cuda(:\d+)?)$")
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


class ConfigError(ValueError):
    """Raised when configuration is missing or invalid."""


@dataclass
class ModelConfig:
    id: str = "Qwen/Qwen3-0.6B"
    cache_dir: Path = PROJECT_ROOT / "models"
    device: str = "auto"
    dtype: str = "auto"
    # A saved LoRA adapter directory merged into the base model by the
    # answering pipeline (regulus.rag.load_qwen_generator), or None for the
    # base model. `model.adapter` in the YAML, REGULUS_ADAPTER in the
    # environment ("none" or empty clears it).
    adapter: Path | None = None
    # Stage 7: where generation runs. "transformers" = in-process (default); "openai_compatible" = an HTTP
    # server speaking the OpenAI chat-completions API (vLLM, TGI, llama.cpp, MLX-LM: remote or quantized
    # models) at `endpoint`. The API key, when needed, is read from REGULUS_MODEL_API_KEY only.
    backend: str = "transformers"
    endpoint: str | None = None
    # "mlx" (opt-in, Apple Silicon): generation runs in a separate worker process under the MLX runtime
    # (regulus.rag.mlx_generator). The worker's interpreter (an environment with mlx-lm installed, kept
    # apart from this one) and the MLX model directory are explicit; `id`/`cache_dir` still name the
    # tokenizer and chat template, which stay in this process. `mlx_revision` records which checkpoint
    # revision the directory holds. `mlx_min_available_mb` is the memory preflight: the worker is not
    # started when less memory is available (0 disables the check).
    mlx_python: Path | None = None
    mlx_model_path: Path | None = None
    mlx_revision: str | None = None
    mlx_min_available_mb: int = 2000
    # What runs the worker's forward passes over that same MLX model directory: "mlx" (Apple Silicon, default) or
    # "torch" (scripts/torch_worker.py: Windows/Linux, CUDA or CPU). Prompt, budget, repetition stop and decoding
    # stay in this process either way. With "torch", mlx_python defaults to this interpreter (it has torch).
    worker_runtime: str = "mlx"

    def validate(self) -> None:
        if self.backend not in ("transformers", "openai_compatible", "mlx"):
            raise ConfigError(f"Invalid backend {self.backend!r}; use transformers, openai_compatible or mlx")
        if self.worker_runtime not in ("mlx", "torch"):
            raise ConfigError(f"Invalid worker runtime {self.worker_runtime!r}; use mlx or torch")
        if self.backend == "openai_compatible" and not self.endpoint:
            raise ConfigError("backend openai_compatible needs model.endpoint (or REGULUS_MODEL_ENDPOINT)")
        if self.backend == "mlx" and (self.mlx_python is None or self.mlx_model_path is None):
            raise ConfigError("backend mlx needs model.mlx_python and model.mlx_model_path "
                              "(or REGULUS_MLX_PYTHON and REGULUS_MLX_MODEL_PATH)")
        if self.mlx_min_available_mb < 0:
            raise ConfigError("model.mlx_min_available_mb must be >= 0")
        if not _MODEL_ID_RE.match(self.id):
            raise ConfigError(
                f"Invalid model id {self.id!r}; expected '<org>/<name>' (e.g. Qwen/Qwen3-0.6B)"
            )
        if not _DEVICE_RE.match(self.device):
            raise ConfigError(f"Invalid device {self.device!r}; use auto, cpu, mps, cuda or cuda:N")
        if self.dtype not in VALID_DTYPES:
            raise ConfigError(f"Invalid dtype {self.dtype!r}; use one of {sorted(VALID_DTYPES)}")


@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    do_sample: bool = True
    enable_thinking: bool = False
    # Stage 4: repetition controls (1.0 / 0 = off, the transformers defaults)
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0


@dataclass
class AnswerLengthConfig:
    """How long a grounded answer is allowed to be - which is not how long it
    should be.

    `policy="adaptive"` lets each question's allowance follow the evidence it
    is grounded in. `max_new_tokens` is a **safety ceiling** against a model
    that never stops, not a target length, and is deliberately generous; see
    docs/answer_length_policy.md before lowering it. Nothing here instructs
    the model to be brief: that is the system prompt's job, so that brevity
    can never cost completeness.
    """

    policy: str = "adaptive"
    max_new_tokens: int = 2048      # the hard ceiling
    min_tokens: int = 256
    base_tokens: int = 96
    per_unit_tokens: int = 160
    range_bonus_tokens: int = 64
    general_tokens: int = 384
    max_units: int = 10
    evidence_multiple: float = 1.4
    evidence_cap_fraction: float = 0.75

    def validate(self) -> None:
        if self.policy not in ("adaptive", "fixed"):
            raise ConfigError(f"answer_length.policy must be 'adaptive' or 'fixed', got {self.policy!r}")
        if self.max_new_tokens <= 0 or self.min_tokens <= 0:
            raise ConfigError("answer_length token values must be positive")
        if self.min_tokens > self.max_new_tokens:
            raise ConfigError(
                f"answer_length.min_tokens ({self.min_tokens}) cannot exceed max_new_tokens ({self.max_new_tokens})"
            )
        if self.evidence_multiple <= 0:
            raise ConfigError("answer_length.evidence_multiple must be positive")
        if not 0 < self.evidence_cap_fraction <= 1:
            raise ConfigError("answer_length.evidence_cap_fraction must be in (0, 1]")


@dataclass
class Settings:
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    answer_length: AnswerLengthConfig = field(default_factory=AnswerLengthConfig)
    source: Path | None = None


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} must contain a mapping at top level")
    return data


def _env(name: str) -> str | None:
    value = os.environ.get(ENV_PREFIX + name)
    return value.strip() if value else None


def _resolve_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _optional_path(raw: Any) -> Path | None:
    return None if raw in (None, "") else _resolve_path(raw)


def _adapter_path(raw: Any) -> Path | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() in ("none", "null", "base"):
        return None
    return _resolve_path(text)


def load_settings(config_path: str | Path | None = None, *, use_dotenv: bool = True) -> Settings:
    """Load settings from YAML, then apply REGULUS_* environment overrides."""
    if use_dotenv:
        load_dotenv(PROJECT_ROOT / ".env", override=False)

    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    data = _read_yaml(path)
    model_data = data.get("model", {}) or {}
    gen_data = data.get("generation", {}) or {}

    worker_runtime = (_env("WORKER_RUNTIME") or str(model_data.get("worker_runtime", ModelConfig.worker_runtime))).lower()
    model = ModelConfig(
        id=_env("MODEL_ID") or str(model_data.get("id", ModelConfig.id)),
        cache_dir=_resolve_path(_env("MODEL_CACHE_DIR") or model_data.get("cache_dir", "models")),
        device=(_env("DEVICE") or str(model_data.get("device", "auto"))).lower(),
        dtype=(_env("DTYPE") or str(model_data.get("dtype", "auto"))).lower(),
        adapter=_adapter_path(os.environ[ENV_PREFIX + "ADAPTER"] if ENV_PREFIX + "ADAPTER" in os.environ
                              else model_data.get("adapter")),
        backend=(_env("MODEL_BACKEND") or str(model_data.get("backend", "transformers"))).lower(),
        endpoint=_env("MODEL_ENDPOINT") or model_data.get("endpoint"),
        mlx_python=_optional_path(_env("MLX_PYTHON") or model_data.get("mlx_python")
                                  or (sys.executable if worker_runtime == "torch" else None)),
        mlx_model_path=_optional_path(_env("MLX_MODEL_PATH") or model_data.get("mlx_model_path")),
        mlx_revision=_env("MLX_REVISION") or model_data.get("mlx_revision"),
        mlx_min_available_mb=int(_env("MLX_MIN_AVAILABLE_MB") or model_data.get("mlx_min_available_mb",
                                                                               ModelConfig.mlx_min_available_mb)),
        worker_runtime=worker_runtime,
    )
    model.validate()

    defaults = GenerationConfig()
    generation = GenerationConfig(
        max_new_tokens=int(gen_data.get("max_new_tokens", defaults.max_new_tokens)),
        temperature=float(gen_data.get("temperature", defaults.temperature)),
        top_p=float(gen_data.get("top_p", defaults.top_p)),
        top_k=int(gen_data.get("top_k", defaults.top_k)),
        do_sample=bool(gen_data.get("do_sample", defaults.do_sample)),
        enable_thinking=bool(gen_data.get("enable_thinking", defaults.enable_thinking)),
        repetition_penalty=float(gen_data.get("repetition_penalty", defaults.repetition_penalty)),
        no_repeat_ngram_size=int(gen_data.get("no_repeat_ngram_size", defaults.no_repeat_ngram_size)),
    )

    length_data = data.get("answer_length", {}) or {}
    length_defaults = AnswerLengthConfig()
    answer_length = AnswerLengthConfig(
        policy=(_env("ANSWER_LENGTH_POLICY") or str(length_data.get("policy", length_defaults.policy))).lower(),
        max_new_tokens=int(_env("MAX_NEW_TOKENS") or length_data.get("max_new_tokens", length_defaults.max_new_tokens)),
        min_tokens=int(length_data.get("min_tokens", length_defaults.min_tokens)),
        base_tokens=int(length_data.get("base_tokens", length_defaults.base_tokens)),
        per_unit_tokens=int(length_data.get("per_unit_tokens", length_defaults.per_unit_tokens)),
        range_bonus_tokens=int(length_data.get("range_bonus_tokens", length_defaults.range_bonus_tokens)),
        general_tokens=int(length_data.get("general_tokens", length_defaults.general_tokens)),
        max_units=int(length_data.get("max_units", length_defaults.max_units)),
        evidence_multiple=float(length_data.get("evidence_multiple", length_defaults.evidence_multiple)),
        evidence_cap_fraction=float(length_data.get("evidence_cap_fraction", length_defaults.evidence_cap_fraction)),
    )
    answer_length.validate()

    return Settings(model=model, generation=generation, answer_length=answer_length, source=path)
