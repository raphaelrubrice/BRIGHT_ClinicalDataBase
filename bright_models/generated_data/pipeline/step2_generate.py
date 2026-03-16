"""Step 2: Document generation with inline annotations.

For each (profile, doc_type) pair, calls the LLM to produce a JSON containing
both the document text and annotation anchors ({value, span} pairs).
"""

import json
import logging
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Prompt building
# ═════════════════════════════════════════════════════════════════════════════

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_few_shot(doc_type: str, few_shot_dir: Path) -> str:
    """Load few-shot examples for a document type."""
    from config.fields import FEW_SHOT_MAP

    filenames = FEW_SHOT_MAP.get(doc_type, [])
    if not filenames:
        return "(Pas d'exemples disponibles pour ce type de document.)"

    parts = []
    for fname in filenames:
        path = few_shot_dir / fname
        if path.exists():
            parts.append(f"--- Exemple: {fname} ---\n{_read_text(path)}")
        else:
            logger.warning("Few-shot file not found: %s", path)
    return "\n\n".join(parts) if parts else "(Exemples non trouvés.)"


def _filter_profile_for_doc(profile: dict, doc_type: str) -> dict:
    """Keep only fields relevant to this document type + meta fields."""
    from config.fields import DOC_TYPE_FIELDS

    relevant = set(DOC_TYPE_FIELDS.get(doc_type, []))
    meta = {"patient_id", "document_types"}
    filtered = {}
    for k, v in profile.items():
        if v is None:
            continue
        if k in relevant or k in meta:
            filtered[k] = v
    return filtered


def build_prompt(
    doc_type: str,
    profile: dict,
    few_shot_dir: Path,
    prompts_dir: Path,
) -> tuple[str, str]:
    """Build (system, user) prompt pair for a single document generation.

    Returns
    -------
    (system_message, user_message)
    """
    # Load templates
    system_common = _read_text(prompts_dir / "system_common.txt")
    doc_template = _read_text(prompts_dir / f"{doc_type}.txt")

    # Few-shot examples
    few_shot_text = load_few_shot(doc_type, few_shot_dir)

    # Filter profile to relevant fields
    filtered_profile = _filter_profile_for_doc(profile, doc_type)
    profile_json = json.dumps(filtered_profile, ensure_ascii=False, indent=2)

    # Fill the doc-type template
    user_message = doc_template.replace("{system_common}", system_common)
    user_message = user_message.replace("{patient_profile_json}", profile_json)
    user_message = user_message.replace("{few_shot_examples}", few_shot_text)

    # System message: the doc-type persona is the first line of the template
    first_line = doc_template.split("\n")[0]
    system_message = first_line

    return system_message, user_message


# ═════════════════════════════════════════════════════════════════════════════
# Validation of raw LLM output
# ═════════════════════════════════════════════════════════════════════════════

def validate_raw_document(doc: dict) -> bool:
    """Check that a parsed LLM response has the required structure."""
    if not isinstance(doc, dict):
        return False
    if "document_text" not in doc or not isinstance(doc["document_text"], str):
        return False
    if "annotations" not in doc or not isinstance(doc["annotations"], dict):
        return False
    if len(doc["document_text"]) < 100:
        return False
    # Each annotation should have value and span
    for field, ann in doc["annotations"].items():
        if not isinstance(ann, dict):
            return False
        if "value" not in ann or "span" not in ann:
            return False
    return True


# ═════════════════════════════════════════════════════════════════════════════
# Generation loop
# ═════════════════════════════════════════════════════════════════════════════

def _build_task_list(profiles: list[dict]) -> list[tuple[dict, str]]:
    """Build flat list of (profile, doc_type) tasks."""
    tasks = []
    for profile in profiles:
        for doc_type in profile.get("document_types", []):
            tasks.append((profile, doc_type))
    return tasks


def generate_documents(
    profiles: list[dict],
    config,
    llm_client,
    checkpoint_mgr,
) -> list[dict]:
    """Generate documents for all profiles, with checkpointing.

    Parameters
    ----------
    profiles : list[dict]
        Patient profiles (from merged batch_*.json files).
    config : PipelineConfig
        Pipeline configuration.
    llm_client : LLMClient
        Initialised LLM client.
    checkpoint_mgr : CheckpointManager
        For save/resume.

    Returns
    -------
    list[dict]
        All generated documents (including previously checkpointed ones).
    """
    tasks = _build_task_list(profiles)
    logger.info("Total generation tasks: %d", len(tasks))

    # Resume from checkpoint
    skip_count, existing_docs = checkpoint_mgr.load_progress("step2")
    if skip_count > 0:
        logger.info("Resuming from checkpoint: skipping %d tasks", skip_count)
    remaining_tasks = tasks[skip_count:]

    all_docs = list(existing_docs)
    batch_size = config.batch_size
    batch_id = checkpoint_mgr.get_next_batch_id("step2")

    # Use batched generation for local models
    use_batch = (config.llm_provider == "local")

    for i in tqdm(range(0, len(remaining_tasks), batch_size),
                  desc="Generating documents", unit="batch"):
        batch_tasks = remaining_tasks[i:i + batch_size]
        batch_docs = []

        if use_batch:
            # Build all prompts for this batch, then generate in parallel
            prompts = []
            for profile, doc_type in batch_tasks:
                system, user = build_prompt(
                    doc_type, profile, config.few_shot_dir, config.prompts_dir,
                )
                prompts.append((system, user))

            raw_outputs = llm_client.generate_batch(prompts)

            for (profile, doc_type), raw in zip(batch_tasks, raw_outputs):
                doc = _process_output(raw, profile, doc_type, llm_client)
                if doc is not None:
                    batch_docs.append(doc)
        else:
            # Sequential generation (API path)
            for profile, doc_type in batch_tasks:
                system, user = build_prompt(
                    doc_type, profile, config.few_shot_dir, config.prompts_dir,
                )

                try:
                    raw = llm_client.generate(system, user)
                except Exception as e:
                    logger.error(
                        "LLM call failed for %s/%s: %s",
                        profile.get("patient_id"), doc_type, e,
                    )
                    continue

                doc = _process_output(raw, profile, doc_type, llm_client)
                if doc is not None:
                    batch_docs.append(doc)

        # Checkpoint this batch
        if batch_docs:
            checkpoint_mgr.save_batch("step2", batch_docs, batch_id)
            all_docs.extend(batch_docs)
            batch_id += 1

    logger.info(
        "Generation complete: %d documents (%d failed)",
        len(all_docs), len(tasks) - len(all_docs),
    )
    return all_docs


def _process_output(raw: str, profile: dict, doc_type: str, llm_client) -> dict | None:
    """Parse and validate a single LLM output."""
    from pipeline.llm_client import LLMClient

    parsed = LLMClient.parse_json_response(raw)
    if parsed is None:
        logger.warning(
            "JSON parse failed for %s/%s",
            profile.get("patient_id"), doc_type,
        )
        return None

    if not validate_raw_document(parsed):
        logger.warning(
            "Validation failed for %s/%s",
            profile.get("patient_id"), doc_type,
        )
        return None

    # Attach metadata
    parsed["patient_id"] = profile.get("patient_id", "unknown")
    parsed["doc_type"] = doc_type
    return parsed
