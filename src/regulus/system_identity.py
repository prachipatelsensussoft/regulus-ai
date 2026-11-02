"""What system is configured to run right now: one definition for everyone.

The answering UI (`scripts/serve_ui.py`), Stage 5 (`regulus.stage5`) and the
read-only dashboard all build the pipeline configuration through this module,
from the one settings loader (`regulus.config.load_settings`: configs/model.yaml
plus REGULUS_* overrides). That is what makes "this evaluation's fingerprint
equals the current fingerprint" mean "this evaluation measured the pipeline
that is configured today".

Importing this module loads no model and no torch: the dashboard uses it.

The evaluation protocol
-----------------------
Stage 5 runs the configured pipeline with two fixed protocol choices, which
are therefore part of every fingerprint on both sides of the comparison:

* validation_behavior = "annotate": the raw answer is kept and the repaired
  one is derived from it with the pipeline's own deterministic repair, so both
  are measured. The final answer is the one production's "repair" mode shows.
* greedy decoding with thinking off, which is also how the answering UI runs.

Known gaps
----------
* The grounding gate is a per-request option of the answering UI, not
  configuration. Stage 5 never enables it and it is not fingerprinted.
* Pipeline code is not fingerprinted (the project has no version control);
  the system prompt text is, by hash.
* `auto` device and dtype are decided when the model loads (CUDA if available,
  else CPU / float32). Resolving them needs torch, so the dashboard reports
  them as configured. Neither is part of the model key or the fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from regulus.config import GenerationConfig, Settings, load_settings
from regulus.dashboard.fingerprint import normalize_evaluation_config, pipeline_fingerprint
from regulus.rag.generation import GROUNDED_SYSTEM_PROMPT, GenerationBudget
from regulus.rag.pipeline import RagConfig
from regulus.retrieval import lexical as retrieval_lexical

CHECKPOINT_RE = re.compile(r"checkpoint-\d+")
EVALUATION_VALIDATION_BEHAVIOR = "annotate"


# ------------------------------------------------------- pipeline configuration


def _env_choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    """A string switch read like settings._env: stripped, lower-cased, empty = unset; anything else stops start-up."""
    import os

    value = (os.environ.get(name) or "").strip().lower() or default
    if value not in allowed:
        raise ValueError(f"{name}={value!r} is not one of {', '.join(allowed)}")
    return value


def configured_rag_config(settings: Settings) -> RagConfig:
    """The RAG configuration the answering pipeline runs with.

    Section promotion and the answer-first format are on (22 Oct 2026, regulus.rag.section_promotion and
    answer_plan.APPLY_FORMAT); REGULUS_SECTION_PROMOTION (off|lexical|picker) and REGULUS_ANSWER_STYLE
    (restate|apply) override them. An invalid value raises ValueError (serve_ui checks this before it listens)."""
    return RagConfig(generation_budget=GenerationBudget.from_config(settings.answer_length),
                     section_promotion=_env_choice("REGULUS_SECTION_PROMOTION", "picker", ("off", "lexical", "picker")),
                     answer_style=_env_choice("REGULUS_ANSWER_STYLE", "apply", ("restate", "apply")))


def configured_generation_config(settings: Settings) -> GenerationConfig:
    """Decoding as the answering pipeline runs it: greedy, thinking off.

    `max_new_tokens` is set to the answer-length ceiling; the pipeline passes
    its own per-question allowance on every call, so the value is never the
    one that decides.
    """
    import os

    budget = GenerationBudget.from_config(settings.answer_length)
    # REGULUS_REPETITION_PENALTY (22 Oct 2026; the Qwen3-4B-Instruct-2507 profile sets 1.1): unset/empty = configured value
    raw = (os.environ.get("REGULUS_REPETITION_PENALTY") or "").strip()
    try:
        penalty = float(raw) if raw else float(settings.generation.repetition_penalty)
    except ValueError:
        raise ValueError(f"REGULUS_REPETITION_PENALTY={raw!r} is not a number") from None
    if not (1.0 <= penalty <= 2.0):
        raise ValueError(f"REGULUS_REPETITION_PENALTY={penalty} must be between 1.0 and 2.0")
    return replace(settings.generation, max_new_tokens=budget.hard_max_tokens, do_sample=False, enable_thinking=False,
                   repetition_penalty=penalty)


def evaluation_rag_config(settings: Settings) -> RagConfig:
    """The configured pipeline under the Stage 5 protocol (annotate, then repair)."""
    return replace(configured_rag_config(settings), validation_behavior=EVALUATION_VALIDATION_BEHAVIOR)


def pipeline_config(config: RagConfig) -> dict[str, Any]:
    """The model-independent sections of the evaluation configuration."""
    rag = config.to_dict()
    retrieval = {k: rag.get(k) for k in ("retriever", "top_k", "corpus", "min_trigram_similarity", "exclude_contextual_only", "semantic_fallback")}
    if retrieval_lexical.ECFR_LITERAL_OR_ENABLED:
        # general questions' eCFR all-words search without the "or" operator (default on since 22 Oct; =0 turns it
        # off and restores the earlier fingerprint)
        retrieval["ecfr_literal_or"] = True
    return {
        "retrieval": retrieval,
        "coverage": {k: rag.get(k) for k in ("no_context_behavior", "incomplete_behavior", "answerability", "optional_context", "scope_policy", "evidence_consolidation")},
        "context": dict(rag.get("context") or {}),
        "generation_budget": dict(rag.get("generation_budget") or {}),
        "validation": {"validation_behavior": rag.get("validation_behavior")},
        # Stage 5 keeps the raw answer and derives the repaired one with the
        # pipeline's own deterministic repair function.
        "repair": {"mode": "post_hoc_deterministic"},
        "prompt": {"system_prompt_sha256": hashlib.sha256(GROUNDED_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
                   # Stage 12: whether the user turn carries the answer shape and evidence plan
                   "answer_shape": rag.get("answer_shape")},
        # 22 Oct 2026: recorded only when on, so the fingerprints of earlier configurations are unchanged
        **({"promotion": {"section_promotion": config.section_promotion, "statute_store_sha256": _statute_store_sha()}}
           if config.section_promotion != "off" else {}),
        **({"answer_style": {"answer_style": config.answer_style,
                             "apply_format_sha256": hashlib.sha256(_apply_format().encode("utf-8")).hexdigest()}}
           if config.answer_style != "restate" else {}),
    }


def _statute_store_sha() -> str | None:
    from regulus.retrieval.statute import STORE

    return hashlib.sha256(STORE.read_bytes()).hexdigest() if STORE.exists() else None


def _apply_format() -> str:
    from regulus.rag.answer_plan import apply_format
    from regulus.rag.generation import APPLY_RETRY_SYSTEM_PROMPT

    return apply_format() + "\n" + APPLY_RETRY_SYSTEM_PROMPT   # DA1 on: its wording is part of the identity


def decoding_config(generation: GenerationConfig) -> dict[str, Any]:
    return {
        "do_sample": generation.do_sample,
        "temperature": generation.temperature,
        "top_p": generation.top_p,
        "top_k": generation.top_k,
        "enable_thinking": generation.enable_thinking,
        "repetition_penalty": generation.repetition_penalty,
        "no_repeat_ngram_size": generation.no_repeat_ngram_size,
    }


def evaluation_config(config: RagConfig, generation: GenerationConfig | None) -> dict[str, Any]:
    """The evaluation-relevant configuration, normalised for fingerprinting.

    `generation` is None before a model is loaded; the decoding section is
    then recorded as absent rather than guessed.
    """
    raw = pipeline_config(config)
    raw["decoding"] = decoding_config(generation) if generation is not None else None
    return normalize_evaluation_config(raw)


# ------------------------------------------------------------------- adapters


class AdapterError(RuntimeError):
    """A path that should be a saved adapter is not one (re-exported by regulus.finetune)."""


def adapter_identity(adapter_dir: Path) -> dict[str, Any]:
    """Facts about a saved adapter: base model, LoRA config, training metadata if present.

    Reads two small JSON files and lists a directory; loads nothing.
    `regulus.finetune.adapter_identity` is this function.
    """
    adapter_dir = Path(adapter_dir)
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.is_file():
        raise AdapterError(f"no adapter_config.json in {adapter_dir.name}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    weights = [p.name for p in adapter_dir.iterdir() if p.name.startswith("adapter_model.")]
    identity = {
        "adapter_dir": adapter_dir.as_posix(),
        "adapter_name": adapter_dir.parent.name if adapter_dir.name == "adapter" else adapter_dir.name,
        "base_model_name_or_path": cfg.get("base_model_name_or_path"),
        "peft_type": cfg.get("peft_type"),
        "r": cfg.get("r"),
        "lora_alpha": cfg.get("lora_alpha"),
        "lora_dropout": cfg.get("lora_dropout"),
        "target_modules": sorted(cfg.get("target_modules") or []),
        "weights": weights,
    }
    meta = adapter_dir.parent / "metadata.json"
    if meta.is_file():
        m = json.loads(meta.read_text(encoding="utf-8"))
        identity["stage"] = m.get("stage")
        identity["dataset_hash"] = (m.get("dataset") or {}).get("hash")
        identity["best_checkpoint"] = m.get("best_checkpoint")
        identity["timestamp"] = m.get("timestamp")
    return identity


def resolve_adapter(identity: dict[str, Any] | None) -> dict[str, Any]:
    """Name the adapter and the exact checkpoint whose weights are loaded.

    The checkpoint is taken only from a fact, never from directory order or
    timestamps:

    * an adapter path that is itself `.../checkpoint-N` names its checkpoint;
    * a training run's `adapter/` directory holds the weights the trainer
      reloaded before saving (load_best_model_at_end), which `metadata.json`
      names as `best_checkpoint`.

    Anything else leaves the checkpoint null: the weights are then unresolved
    rather than every checkpoint of the adapter collapsing into one identity.
    """
    if not identity:
        return {}
    path = Path(str(identity.get("adapter_dir") or ""))
    if CHECKPOINT_RE.fullmatch(path.name):
        run_dir = path.parent.parent if path.parent.name == "checkpoints" else path.parent
        return {"adapter_name": run_dir.name, "checkpoint": path.name,
                "checkpoint_source": "the adapter path is the checkpoint directory"}
    if path.name == "adapter" and identity.get("best_checkpoint"):
        return {"adapter_name": path.parent.name, "checkpoint": identity["best_checkpoint"],
                "checkpoint_source": "metadata.json best_checkpoint of the training run"}
    return {"adapter_name": identity.get("adapter_name") or path.name or None, "checkpoint": None,
            "checkpoint_source": None}


# ------------------------------------------------------------------ revision


def cached_revision(model_id: str, cache_dir: Path) -> tuple[str | None, str | None]:
    """The commit the model loader will load, read from the local HF cache.

    The loaders pass no revision, so transformers loads `main`, which in the
    cache is the commit named by `refs/main`. Returns (revision, None) or
    (None, why it could not be resolved).
    """
    repo = Path(cache_dir) / f"models--{model_id.replace('/', '--')}"
    ref = repo / "refs" / "main"
    if not ref.is_file():
        return None, f"{ref.as_posix()} does not exist, so the base-model revision the loader would use is unknown"
    revision = ref.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        return None, f"{ref.as_posix()} does not hold a commit hash"
    if not (repo / "snapshots" / revision).is_dir():
        return None, f"refs/main names {revision[:12]}, but that snapshot is not in the cache"
    return revision, None


# ------------------------------------------------------------ the whole system


@dataclass(frozen=True)
class ConfiguredSystem:
    """Plain data: what the configuration says the system runs, fully resolved
    where that is possible without loading anything, and why not otherwise."""

    model: dict[str, Any]
    evaluation_config: dict[str, Any] | None
    pipeline_fingerprint: dict[str, Any] | None
    unresolved: tuple[str, ...] = ()
    source: dict[str, Any] = field(default_factory=dict)

    @property
    def resolved(self) -> bool:
        return not self.unresolved

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "unresolved": list(self.unresolved), "resolved": self.resolved}


def describe_settings(settings: Settings) -> ConfiguredSystem:
    """Resolve the configured system from loaded settings."""
    m = settings.model
    unresolved: list[str] = []
    revision, why = cached_revision(m.id, m.cache_dir)
    if why:
        unresolved.append(why)

    adapter: dict[str, Any] = {}
    info: dict[str, Any] | None = None
    if m.adapter is not None:
        try:
            info = adapter_identity(m.adapter)
        except Exception as exc:  # the configuration names something that is not an adapter
            unresolved.append(f"the configured adapter {Path(m.adapter).as_posix()} cannot be read: {exc}")
        if info is not None:
            adapter = resolve_adapter(info)
            base = info.get("base_model_name_or_path")
            if base and base != m.id:
                unresolved.append(f"the configured adapter was trained on {base}, not on the configured base model {m.id}")
            if not adapter.get("checkpoint"):
                unresolved.append(
                    f"the configured adapter {adapter.get('adapter_name')} does not say which checkpoint its weights "
                    "are, so the weights cannot be identified"
                )
        else:
            adapter = {"adapter_name": Path(m.adapter).parent.name if Path(m.adapter).name == "adapter" else Path(m.adapter).name,
                       "checkpoint": None, "checkpoint_source": None}

    model = {
        "id": m.id,
        "revision": revision,
        "adapter_dir": Path(m.adapter).as_posix() if m.adapter is not None else None,
        "adapter_name": adapter.get("adapter_name"),
        "checkpoint": adapter.get("checkpoint"),
        "checkpoint_source": adapter.get("checkpoint_source"),
        "device_configured": m.device,
        "dtype_configured": m.dtype,
        "device": m.device if m.device != "auto" else None,
        "dtype": m.dtype if m.dtype != "auto" else None,
    }
    econfig = evaluation_config(evaluation_rag_config(settings), configured_generation_config(settings))
    return ConfiguredSystem(
        model=model,
        evaluation_config=econfig,
        pipeline_fingerprint=pipeline_fingerprint(econfig).to_dict(),
        unresolved=tuple(unresolved),
        source={
            "config_file": Path(settings.source).as_posix() if settings.source else None,
            "loader": "regulus.config.load_settings (configs/model.yaml, then REGULUS_* environment overrides)",
            "revision_from": "the local Hugging Face cache refs/main (what the loader resolves)",
            "pipeline_from": "regulus.system_identity.evaluation_rag_config and configured_generation_config",
            "runtime_note": ("device and dtype set to auto are decided when the model loads; they are not part of "
                             "the model identity key or the pipeline fingerprint"),
        },
    )


def configured_system(config_path: str | Path | None = None, *, use_dotenv: bool = True) -> ConfiguredSystem:
    return describe_settings(load_settings(config_path, use_dotenv=use_dotenv))
