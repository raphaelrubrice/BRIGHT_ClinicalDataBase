"""Checkpoint save/resume for Colab resilience.

Each batch is written as an append-only JSONL file. If a Colab session dies
mid-batch, at most one batch (~20 documents) is lost. On resume,
load_progress() counts completed items so the generation loop can skip them.
"""

import json
import logging
from pathlib import Path

try:
    import orjson

    def _dumps(obj: dict) -> str:
        return orjson.dumps(obj, option=orjson.OPT_NON_STR_KEYS).decode()
except ImportError:
    def _dumps(obj: dict) -> str:
        return json.dumps(obj, ensure_ascii=False)

logger = logging.getLogger(__name__)


class CheckpointManager:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)

    def save_batch(self, step: str, items: list[dict], batch_id: int) -> Path:
        """Write a batch of items as JSONL. Returns the path written."""
        step_dir = self.checkpoint_dir / step
        step_dir.mkdir(parents=True, exist_ok=True)
        path = step_dir / f"batch_{batch_id:04d}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(_dumps(item) + "\n")
        logger.info("Saved %d items to %s", len(items), path)
        return path

    def load_progress(self, step: str) -> tuple[int, list[dict]]:
        """Load all completed items for a step.

        Returns (count, list_of_items).
        """
        step_dir = self.checkpoint_dir / step
        if not step_dir.exists():
            return 0, []

        items: list[dict] = []
        for path in sorted(step_dir.glob("batch_*.jsonl")):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        items.append(json.loads(line))
        logger.info("Loaded %d items from %s checkpoints", len(items), step)
        return len(items), items

    def get_next_batch_id(self, step: str) -> int:
        """Return the next batch ID for a step."""
        step_dir = self.checkpoint_dir / step
        if not step_dir.exists():
            return 0
        existing = sorted(step_dir.glob("batch_*.jsonl"))
        if not existing:
            return 0
        last = existing[-1].stem  # "batch_0042"
        return int(last.split("_")[1]) + 1

    @staticmethod
    def is_colab() -> bool:
        """Detect if running in Google Colab."""
        try:
            import google.colab  # noqa: F401
            return True
        except ImportError:
            return False
