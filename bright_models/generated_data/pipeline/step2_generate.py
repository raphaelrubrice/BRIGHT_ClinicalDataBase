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
    """Load few-shot examples for a document type.

    JSON files are loaded and re-serialised as compact JSON so the model
    sees a complete input/output example.  Plain-text files are included
    verbatim (legacy support).
    """
    from config.fields import FEW_SHOT_MAP

    filenames = FEW_SHOT_MAP.get(doc_type, [])
    if not filenames:
        return "(Pas d'exemples disponibles pour ce type de document.)"

    parts = []
    for fname in filenames:
        path = few_shot_dir / fname
        if not path.exists():
            logger.warning("Few-shot file not found: %s", path)
            continue
        if path.suffix == ".json":
            # JSON few-shot: show the model what a complete output looks like
            raw = _read_text(path)
            try:
                obj = json.loads(raw)
                parts.append(
                    f"--- Exemple de sortie JSON attendue ---\n"
                    f"{json.dumps(obj, ensure_ascii=False, indent=2)}"
                )
            except json.JSONDecodeError:
                logger.warning("Invalid JSON in few-shot file: %s", path)
                parts.append(f"--- Exemple: {fname} ---\n{raw}")
        else:
            parts.append(f"--- Exemple: {fname} ---\n{raw}")
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

            failed_retries = []
            for (profile, doc_type), (system, user), raw in zip(batch_tasks, prompts, raw_outputs):
                doc = _process_output(raw, profile, doc_type, llm_client)
                if doc is not None:
                    batch_docs.append(doc)
                else:
                    failed_retries.append((profile, doc_type, system, user))

            # Retry failed items individually (fresh random seed)
            for profile, doc_type, system, user in failed_retries:
                try:
                    raw = llm_client.generate(system, user)
                    doc = _process_output(raw, profile, doc_type, llm_client)
                    if doc is not None:
                        batch_docs.append(doc)
                        logger.info("Retry succeeded for %s/%s",
                                    profile.get("patient_id"), doc_type)
                    else:
                        logger.warning("Retry also failed for %s/%s",
                                       profile.get("patient_id"), doc_type)
                except Exception as e:
                    logger.error("Retry error for %s/%s: %s",
                                 profile.get("patient_id"), doc_type, e)
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


def _value_in_span(value: str, span: str) -> bool:
    """Check if the annotation value appears (loosely) inside the span."""
    val_lower = value.lower().strip()
    span_lower = span.lower()
    # Direct substring
    if val_lower in span_lower:
        return True
    # Check individual value tokens (at least half should appear)
    tokens = [t for t in val_lower.split() if len(t) > 2]
    if not tokens:
        return True  # very short value, can't check
    hits = sum(1 for t in tokens if t in span_lower)
    return hits >= max(1, len(tokens) * 0.5)


def _find_value_in_text(value: str, text: str, text_lower: str) -> str | None:
    """Try to find the value in the document text and return a context span.

    Returns a verbatim substring of *text* (original case) or None.
    """
    val_lower = value.lower().strip()
    if len(val_lower) < 2:
        return None

    # 1. Exact substring match (case-insensitive)
    pos = text_lower.find(val_lower)
    if pos >= 0:
        # Expand to ±50 chars for context, snapping to word boundaries
        start = max(0, text_lower.rfind(" ", max(0, pos - 50), pos) + 1)
        end_val = pos + len(val_lower)
        end = text_lower.find(" ", end_val, end_val + 50)
        if end == -1:
            end = min(len(text), end_val + 50)
        return text[start:end].strip()

    # 2. Token-level search: find the first region containing most value tokens
    tokens = [t for t in val_lower.split() if len(t) > 2]
    if not tokens:
        return None
    # Find position of each token
    positions = []
    for tok in tokens:
        p = text_lower.find(tok)
        if p >= 0:
            positions.append(p)
    if len(positions) < max(1, len(tokens) * 0.5):
        return None  # not enough tokens found
    # Build span around the cluster of found tokens
    cluster_start = min(positions)
    cluster_end = max(p + len(tokens[0]) for p in positions)  # approximate
    start = max(0, text_lower.rfind(" ", max(0, cluster_start - 30), cluster_start) + 1)
    end = text_lower.find(" ", cluster_end, cluster_end + 30)
    if end == -1:
        end = min(len(text), cluster_end + 30)
    span = text[start:end].strip()
    return span if len(span) > len(val_lower) else None


def _clean_annotations(doc: dict) -> dict:
    """Post-process annotations: fix field names, repair/drop broken spans.

    1. Fuzzy-map invented field names to the closest ALL_111_FIELDS match.
    2. Verify span exists in document_text (drop if not).
    2b. If value doesn't match span, try to find value in text and repair span.
    3. Keep first occurrence for duplicate resolved names.
    """
    from profiles_validation import ALL_111_FIELDS

    try:
        from rapidfuzz import fuzz, process as rf_process
        _has_rapidfuzz = True
    except ImportError:
        _has_rapidfuzz = False

    valid_fields = set(ALL_111_FIELDS)
    text = doc.get("document_text", "")
    text_lower = text.lower()
    annotations = doc.get("annotations", {})
    cleaned: dict = {}
    stats = {"renamed": 0, "dropped_name": 0, "dropped_span": 0,
             "repaired": 0, "dropped_value": 0, "kept": 0}

    for field, ann in annotations.items():
        if not isinstance(ann, dict) or "value" not in ann or "span" not in ann:
            continue

        # --- Step 1: resolve field name ---
        resolved_name = field
        if field not in valid_fields:
            if _has_rapidfuzz:
                match = rf_process.extractOne(
                    field, ALL_111_FIELDS, scorer=fuzz.ratio, score_cutoff=60,
                )
                if match:
                    resolved_name = match[0]
                    logger.debug("Field rename: %s -> %s (score=%d)", field, resolved_name, match[1])
                    stats["renamed"] += 1
                else:
                    logger.debug("Field dropped (no match): %s", field)
                    stats["dropped_name"] += 1
                    continue
            else:
                logger.debug("Field dropped (unknown, no rapidfuzz): %s", field)
                stats["dropped_name"] += 1
                continue

        # --- Step 2: verify span exists in document_text ---
        span = ann.get("span", "")
        if span and span.lower() not in text_lower:
            logger.debug("Span not found, dropping %s: '%.60s'", resolved_name, span)
            stats["dropped_span"] += 1
            continue

        # --- Step 2b: coerce value/span to string (LLM sometimes returns lists) ---
        value = ann.get("value", "")
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        elif not isinstance(value, str):
            value = str(value)
        if isinstance(span, list):
            span = ", ".join(str(s) for s in span)
        elif not isinstance(span, str):
            span = str(span)
        ann = {**ann, "value": value, "span": span}

        # --- Step 2c: verify value is consistent with span; repair if not ---
        if span and value and not _value_in_span(value, span):
            # Span exists in text but doesn't match the value, try to repair
            repaired = _find_value_in_text(value, text, text_lower)
            if repaired:
                logger.debug("Span repaired for %s: '%.40s' -> '%.40s'",
                             resolved_name, span, repaired)
                ann = {**ann, "span": repaired}
                stats["repaired"] += 1
            else:
                logger.debug("Value not in text, dropping %s: value='%.40s'",
                             resolved_name, value)
                stats["dropped_value"] += 1
                continue

        # --- Step 3: keep (first occurrence wins for duplicate resolved names) ---
        if resolved_name not in cleaned:
            cleaned[resolved_name] = ann
            stats["kept"] += 1

    n_changes = stats["renamed"] + stats["dropped_name"] + stats["dropped_span"] + stats["repaired"] + stats["dropped_value"]
    if n_changes:
        logger.info(
            "Annotation cleanup: kept=%d renamed=%d repaired=%d "
            "dropped(name=%d span=%d value=%d)",
            stats["kept"], stats["renamed"], stats["repaired"],
            stats["dropped_name"], stats["dropped_span"], stats["dropped_value"],
        )

    doc["annotations"] = cleaned
    return doc


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

    # Clean up annotations (fuzzy field name mapping + span verification)
    parsed = _clean_annotations(parsed)

    # Attach metadata
    parsed["patient_id"] = profile.get("patient_id", "unknown")
    parsed["doc_type"] = doc_type
    return parsed
