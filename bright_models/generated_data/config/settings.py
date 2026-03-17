"""Pipeline configuration for synthetic neuro-oncology document generation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Base directory: generated_data/
_BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class PipelineConfig:
    # ── Paths ────────────────────────────────────────────────────────────
    profiles_dir: Path = _BASE_DIR / "profiles"
    few_shot_dir: Path = _BASE_DIR / "data" / "few_shot"
    output_dir: Path = _BASE_DIR / "data"
    checkpoint_dir: Path = _BASE_DIR / "data" / "checkpoints"

    # ── LLM ──────────────────────────────────────────────────────────────
    llm_provider: str = "local"          # "local" | "anthropic" | "openai"
    llm_model: str = "Qwen/Qwen3-8B-Instruct"  # T4 default
    llm_quantization: str = "awq"        # "awq" | "gptq" | "none"
    llm_backend: str = "vllm"            # "vllm" | "transformers"
    hf_token: Optional[str] = None       # HuggingFace token for gated models
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 8192
    temperature: float = 0.8
    max_tokens: int = 4096
    batch_size: int = 20                 # documents per checkpoint batch
    enable_thinking: bool = False        # Qwen3 thinking mode (off for speed)

    # API-only settings (ignored for local)
    api_rate_limit_rpm: int = 50
    api_retry_count: int = 3

    # ── Step 3: Span resolution ──────────────────────────────────────────
    fuzzy_threshold: int = 85
    min_resolution_rate: float = 0.90

    # ── Step 4: Quality filtering ────────────────────────────────────────
    reject_resolution_below: float = 0.80
    review_resolution_below: float = 0.90
    length_limits: dict = field(default_factory=lambda: {
        "consultation": {"min": 300, "max": 4000},
        "rcp": {"min": 400, "max": 5000},
        "anapath": {"min": 500, "max": 6000},
    })
    profile_match_reject: float = 0.70
    profile_match_review: float = 0.85
    dedup_similarity: float = 0.85
    prompt_leakage_terms: list = field(default_factory=lambda: [
        "JSON", "annotation", "span", "field_name", "document_text",
    ])

    # ── Derived paths ────────────────────────────────────────────────────
    @property
    def raw_documents_dir(self) -> Path:
        return self.output_dir / "raw_documents"

    @property
    def resolved_dir(self) -> Path:
        return self.output_dir / "resolved"

    @property
    def final_dir(self) -> Path:
        return self.output_dir / "final"

    @property
    def prompts_dir(self) -> Path:
        return _BASE_DIR / "config" / "prompts"

    # ── Factory methods ──────────────────────────────────────────────────
    @classmethod
    def from_colab(
        cls,
        drive_root: str = "/content/drive/MyDrive/synth_neuro_onco",
        gpu: Optional[str] = None,
    ) -> "PipelineConfig":
        """Config for Google Colab with Drive checkpointing."""
        if gpu is None:
            gpu = _detect_gpu()

        model = ("Qwen/Qwen3-32B-Instruct" if "a100" in gpu.lower()
                 else "Qwen/Qwen3-8B-Instruct")
        mem = 0.92 if "a100" in gpu.lower() else 0.90

        return cls(
            checkpoint_dir=Path(drive_root) / "checkpoints",
            llm_model=model,
            gpu_memory_utilization=mem,
        )

    @classmethod
    def from_local(cls, **overrides) -> "PipelineConfig":
        """Config for local execution with default paths."""
        return cls(**overrides)


def _detect_gpu() -> str:
    """Detect GPU type on Colab."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        )
        return out.strip().lower()
    except Exception:
        return "t4"
