#!/usr/bin/env python3
"""Extend the generated dataset with typos and value variations.

Creates 600 augmented document copies (200 per doc type) from the original
dataset, introducing realistic typos and field value substitutions.
Combines original + augmented into generated_dataset_extended.jsonl.
"""

import copy
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
AUGMENTED_PER_TYPE = 200  # 200 per doc type = 600 total
PATIENT_ID_START = 700    # new patients start at P700

random.seed(SEED)

BASE = Path(__file__).resolve().parent.parent
DATASET = BASE / "data" / "generated_dataset.jsonl"
OUTPUT = BASE / "data" / "generated_dataset_extended.jsonl"

NOTE_ID_RE = re.compile(r"synth-P(\d+)-(\w+)")

# ---------------------------------------------------------------------------
# Value substitution tables
# ---------------------------------------------------------------------------

# Each key maps a normalized original value to a list of possible replacements.
# At substitution time, we pick a random replacement that differs from current.

_CHIMIOS_SYNONYMS = {
    "temozolomide": ["témozolomide", "TMZ", "temodar", "témodal"],
    "témozolomide": ["temozolomide", "TMZ", "temodar", "témodal"],
    "tmz": ["temozolomide", "témodal", "temodar"],
    "témodal": ["temozolomide", "TMZ", "temodar"],
    "temodar": ["temozolomide", "TMZ", "témodal"],
    "bevacizumab": ["bévacizumab", "avastin"],
    "bévacizumab": ["bevacizumab", "avastin"],
    "avastin": ["bevacizumab", "bévacizumab"],
    "lomustine": ["CCNU", "bélustine"],
    "ccnu": ["lomustine", "bélustine"],
    "bélustine": ["lomustine", "CCNU"],
    "belustine": ["lomustine", "CCNU"],
    "carmustine": ["BCNU", "gliadel"],
    "bcnu": ["carmustine", "gliadel"],
    "gliadel": ["carmustine", "BCNU"],
    "vincristine": ["oncovin", "VCR"],
    "oncovin": ["vincristine", "VCR"],
    "carboplatine": ["carboplatin", "carbo"],
    "carboplatin": ["carboplatine", "carbo"],
    "procarbazine": ["natulan"],
    "natulan": ["procarbazine"],
    "levetiracetam": ["keppra", "Lev"],
    "keppra": ["levetiracetam", "lévétiracétam"],
    "lévétiracétam": ["keppra", "levetiracetam"],
    "lacosamide": ["vimpat"],
    "vimpat": ["lacosamide"],
    "valproate": ["dépakine"],
    "dépakine": ["valproate"],
    "depakine": ["valproate"],
    "dexaméthasone": ["dexa", "dectancyl", "DXM"],
    "dexamethasone": ["dexa", "dectancyl", "DXM"],
    "dexa": ["dexaméthasone", "dectancyl"],
    "solumedrol": ["méthylprednisolone"],
    "méthylprednisolone": ["solumedrol"],
    "corticothérapie": ["corticoïdes", "CTC"],
}

_CHIMIO_PROTOCOLE_SUBS = {
    "stupp": ["protocole de Stupp", "Stupp", "STUPP", "protocole Stupp"],
    "protocole de stupp": ["Stupp", "stupp", "protocole Stupp"],
    "protocole stupp": ["Stupp", "protocole de Stupp"],
    "pcv": ["PCV", "protocole PCV"],
    "protocole pcv": ["PCV", "pcv"],
}

_MOL_IDH1_SUBS = {
    # Wildtype variants
    "wt": ["non muté", "sauvage", "IDH1 non muté", "wild-type"],
    "non muté": ["wt", "sauvage", "IDH1 non muté", "wild-type"],
    "sauvage": ["wt", "non muté", "IDH1 non muté"],
    "idh1 non muté": ["wt", "non muté", "sauvage"],
    "wild-type": ["wt", "non muté", "sauvage"],
    # Mutated variants
    "muté (r132h)": ["muté R132H", "mutation R132H", "IDH1 muté R132H"],
    "muté r132h": ["muté (R132H)", "mutation R132H", "IDH1 muté R132H"],
    "mutation r132h": ["muté R132H", "muté (R132H)"],
}

_MOL_MGMT_SUBS = {
    "méthylé": ["methylé", "méthylée", "methylee"],
    "methylé": ["méthylé", "methylee"],
    "non méthylé": ["non methylé", "non methyle", "non méthylée"],
    "non methylé": ["non méthylé", "non methyle"],
    "non methyle": ["non méthylé", "non methylé", "non methylee"],
    "non méthylée": ["non méthylé", "non methylé"],
    "non methylee": ["non méthylé", "non methyle"],
}

_TYPE_CHIRURGIE_SUBS = {
    "exérèse totale": ["résection complète", "exerese totale", "exérèse macroscopiquement complète"],
    "résection complète": ["exérèse totale", "exerese totale"],
    "exerese totale": ["exérèse totale", "résection complète"],
    "exérèse partielle": ["résection partielle", "exerese partielle", "exérèse subtotale"],
    "résection partielle": ["exérèse partielle", "exerese partielle"],
    "exerese partielle": ["exérèse partielle", "résection partielle"],
    "exérèse subtotale": ["résection subtotale", "exerese subtotale", "exérèse partielle"],
    "biopsie": ["biopsie chirurgicale", "biopsie cérébrale"],
    "biopsie stéréotaxique": ["biopsie stereotaxique", "biopsie en conditions stéréotaxiques"],
    "biopsie chirurgicale": ["biopsie", "biopsie cérébrale"],
}

_LATERALITE_SUBS = {
    "gauche": ["à gauche", "G", "gche"],
    "à gauche": ["gauche", "G"],
    "droite": ["à droite", "D", "dte"],
    "à droite": ["droite", "D"],
    "bilatérale": ["bilatéral", "bilateral", "bilatérale"],
}

_POSITION_ACCENT_SUBS = {
    "pariétal": ["parietal"],
    "parietal": ["pariétal"],
    "cérébelleux": ["cerebelleux"],
    "cerebelleux": ["cérébelleux"],
    "fronto-pariétal": ["fronto-parietal", "frontopariétal"],
    "fronto-parietal": ["fronto-pariétal"],
    "temporo-pariétal": ["temporo-parietal", "temporopariétal"],
    "temporo-parietal": ["temporo-pariétal"],
    "pariéto-occipital": ["parieto-occipital", "pariéto occipital"],
    "parieto-occipital": ["pariéto-occipital"],
    "temporal": ["temporo"],
    "frontal": ["fronto"],
    "thalamique": ["thalamus"],
    "thalamus": ["thalamique"],
    "temporo-latéral": ["temporo-lateral", "temporolatéral"],
}

_DIAG_HISTO_SUBS = {
    "glioblastome": ["GBM", "glioblastoma", "glioblastome multiforme"],
    "gbm": ["glioblastome", "glioblastoma"],
    "glioblastoma": ["glioblastome", "GBM"],
    "astrocytome": ["astrocytoma"],
    "astrocytoma": ["astrocytome"],
    "oligodendrogliome": ["oligodendroglioma", "oligo"],
    "oligodendroglioma": ["oligodendrogliome"],
    "oligodendrogliome anaplasique": ["oligodendrogliome anaplasique grade 3", "oligodendroglioma anaplasique"],
    "médulloblastome": ["medulloblastome", "médulloblastoma"],
    "medulloblastome": ["médulloblastome"],
    "épendymome": ["ependymome", "épendymoma"],
    "ependymome": ["épendymome"],
}

# Combine all field-specific substitution tables
_FIELD_SUBS: dict[str, dict[str, list[str]]] = {
    "chimios": _CHIMIOS_SYNONYMS,
    "chimio_protocole": _CHIMIO_PROTOCOLE_SUBS,
    "anti_epileptiques": _CHIMIOS_SYNONYMS,  # reuses drug synonyms
    "corticoides": _CHIMIOS_SYNONYMS,        # reuses drug synonyms
    "mol_idh1": _MOL_IDH1_SUBS,
    "mol_mgmt": _MOL_MGMT_SUBS,
    "type_chirurgie": _TYPE_CHIRURGIE_SUBS,
    "tumeur_lateralite": _LATERALITE_SUBS,
    "tumeur_position": _POSITION_ACCENT_SUBS,
    "diag_histologique": _DIAG_HISTO_SUBS,
}

# Probability of substituting an eligible entity
SUB_PROBABILITY = 0.50

# ---------------------------------------------------------------------------
# Typo injection (outside entity spans)
# ---------------------------------------------------------------------------

_ACCENT_MAP = {
    'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
    'à': 'a', 'â': 'a',
    'ô': 'o', 'ö': 'o',
    'ù': 'u', 'û': 'u', 'ü': 'u',
    'î': 'i', 'ï': 'i',
    'ç': 'c',
    'É': 'E', 'È': 'E', 'Ê': 'E',
    'À': 'A', 'Â': 'A',
    'Ô': 'O', 'Ù': 'U', 'Û': 'U',
    'Î': 'I', 'Ç': 'C',
}

ACCENT_DROP_RATE = 0.10   # 10% of accented chars outside spans
NUM_CHAR_SWAPS = 2        # per doc
NUM_CHAR_DOUBLES = 1      # per doc
NUM_CHAR_DELETIONS = 1    # per doc


def _get_safe_positions(text: str, entities: list[dict]) -> list[int]:
    """Return list of char positions in text that are NOT inside any entity span."""
    occupied = set()
    for e in entities:
        for i in range(e["start"], e["end"]):
            occupied.add(i)
    return [i for i in range(len(text)) if i not in occupied]


def _is_word_char(c: str) -> bool:
    return c.isalpha() or c == '-'


def apply_typos(text: str, entities: list[dict]) -> str:
    """Apply typos to text outside entity spans. Returns modified text with updated spans."""
    safe = set()
    for i in range(len(text)):
        safe.add(i)
    for e in entities:
        for i in range(e["start"], e["end"]):
            safe.discard(i)

    # Convert to list for mutation
    chars = list(text)
    # Track offset shifts for span updates
    # We'll apply typos in reverse order to avoid offset issues

    modifications = []  # (position, type, data)

    # 1. Accent drops
    for i in sorted(safe):
        if i < len(chars) and chars[i] in _ACCENT_MAP:
            if random.random() < ACCENT_DROP_RATE:
                modifications.append((i, 'replace', _ACCENT_MAP[chars[i]]))

    # 2. Char swaps (swap adjacent chars within words)
    word_positions = [i for i in sorted(safe)
                      if i < len(chars) - 1
                      and (i + 1) in safe
                      and _is_word_char(chars[i])
                      and _is_word_char(chars[i + 1])
                      and chars[i] != chars[i + 1]]
    if word_positions:
        for _ in range(min(NUM_CHAR_SWAPS, len(word_positions))):
            pos = random.choice(word_positions)
            modifications.append((pos, 'swap', None))

    # 3. Char doubles
    double_positions = [i for i in sorted(safe)
                        if i < len(chars)
                        and _is_word_char(chars[i])]
    if double_positions:
        for _ in range(min(NUM_CHAR_DOUBLES, len(double_positions))):
            pos = random.choice(double_positions)
            modifications.append((pos, 'double', None))

    # 4. Char deletions
    delete_positions = [i for i in sorted(safe)
                        if i < len(chars)
                        and _is_word_char(chars[i])
                        and i > 0 and i < len(chars) - 1
                        and _is_word_char(chars[i - 1])
                        and _is_word_char(chars[i + 1])]
    if delete_positions:
        for _ in range(min(NUM_CHAR_DELETIONS, len(delete_positions))):
            pos = random.choice(delete_positions)
            modifications.append((pos, 'delete', None))

    # Sort modifications by position descending to apply from end
    modifications.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate positions (only one modification per position)
    seen_pos = set()
    unique_mods = []
    for mod in modifications:
        if mod[0] not in seen_pos:
            seen_pos.add(mod[0])
            unique_mods.append(mod)
    modifications = unique_mods

    # Apply modifications and track offset changes
    for pos, mod_type, data in modifications:
        if mod_type == 'replace':
            chars[pos] = data
            # No offset change
        elif mod_type == 'swap':
            if pos + 1 < len(chars):
                chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
                # No offset change
        elif mod_type == 'double':
            chars.insert(pos, chars[pos])
            # +1 offset for everything after pos
            for e in entities:
                if e["start"] > pos:
                    e["start"] += 1
                if e["end"] > pos:
                    e["end"] += 1
        elif mod_type == 'delete':
            chars.pop(pos)
            # -1 offset for everything after pos
            for e in entities:
                if e["start"] > pos:
                    e["start"] -= 1
                if e["end"] > pos:
                    e["end"] -= 1

    return "".join(chars)


# ---------------------------------------------------------------------------
# Value substitution
# ---------------------------------------------------------------------------

def apply_value_substitutions(text: str, entities: list[dict]) -> str:
    """Apply value substitutions to entities, modifying text and updating spans."""
    # Sort entities by start offset descending (process from end to start)
    indexed_entities = list(enumerate(entities))
    indexed_entities.sort(key=lambda x: x[1]["start"], reverse=True)

    for idx, entity in indexed_entities:
        label = entity["label"]
        if label not in _FIELD_SUBS:
            continue

        if random.random() > SUB_PROBABILITY:
            continue

        subs_table = _FIELD_SUBS[label]
        current_value = entity["value"].lower().strip()

        # Find matching substitution
        replacements = subs_table.get(current_value)
        if not replacements:
            continue

        new_value = random.choice(replacements)

        # Current span in text
        start = entity["start"]
        end = entity["end"]
        old_span_text = text[start:end]
        old_len = end - start
        new_len = len(new_value)
        delta = new_len - old_len

        # Replace in text
        text = text[:start] + new_value + text[end:]

        # Update this entity
        entities[idx]["end"] = start + new_len
        entities[idx]["value"] = new_value

        # Shift all entities that come after this one
        for i, e in enumerate(entities):
            if i == idx:
                continue
            if e["start"] >= end:
                entities[i]["start"] += delta
                entities[i]["end"] += delta
            elif e["start"] > start and e["start"] < end:
                # Overlapping entity starting inside this span, adjust
                # This shouldn't normally happen but handle gracefully
                pass

    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading dataset from {DATASET}")
    docs = []
    with open(DATASET, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    print(f"Loaded {len(docs)} documents")

    # Group by doc type
    by_type: dict[str, list[dict]] = defaultdict(list)
    for doc in docs:
        m = NOTE_ID_RE.match(doc["note_id"])
        if m:
            by_type[m.group(2)].append(doc)

    print(f"Document types: {', '.join(f'{k}={len(v)}' for k, v in by_type.items())}")

    augmented = []
    patient_counter = PATIENT_ID_START
    total_subs = 0
    total_typos = 0

    for dtype in ["consultation", "rcp", "anapath"]:
        pool = by_type.get(dtype, [])
        if not pool:
            print(f"  WARNING: no {dtype} documents found, skipping")
            continue

        samples = random.choices(pool, k=AUGMENTED_PER_TYPE)
        print(f"  Sampling {AUGMENTED_PER_TYPE} from {len(pool)} {dtype} docs")

        for i, src_doc in enumerate(samples):
            # Deep copy to avoid mutating original
            doc = copy.deepcopy(src_doc)

            # Assign new patient ID and note_id
            new_pid = f"P{patient_counter:04d}"
            doc["note_id"] = f"synth-{new_pid}-{dtype}"
            patient_counter += 1

            text = doc["note_text"]
            entities = doc["entities"]

            # Count entities before for stats
            n_ents_before = len(entities)

            # 1. Apply value substitutions (modifies text and entities)
            text = apply_value_substitutions(text, entities)

            # Count how many substitutions happened
            subs_count = sum(
                1 for orig_e, new_e in zip(src_doc["entities"], entities)
                if orig_e["value"] != new_e["value"]
            )
            total_subs += subs_count

            # 2. Apply typos (modifies text and adjusts entity offsets)
            text_before = text
            text = apply_typos(text, entities)
            typo_count = sum(1 for a, b in zip(text_before, text) if a != b)
            # Rough count, not exact due to insertions/deletions
            total_typos += max(typo_count, abs(len(text) - len(text_before)))

            doc["note_text"] = text
            doc["entities"] = entities
            augmented.append(doc)

    print(f"\nAugmentation complete:")
    print(f"  Augmented documents: {len(augmented)}")
    print(f"  Total value substitutions: {total_subs}")
    print(f"  New patient IDs: P{PATIENT_ID_START:04d} - P{patient_counter - 1:04d}")

    # Validate spans on augmented docs
    span_errors = 0
    for doc in augmented:
        text = doc["note_text"]
        text_len = len(text)
        for e in doc["entities"]:
            if e["start"] < 0 or e["end"] > text_len or e["start"] >= e["end"]:
                span_errors += 1
            elif text[e["start"]:e["end"]].strip() == "":
                span_errors += 1

    print(f"  Span errors in augmented docs: {span_errors}")

    # Combine original + augmented
    combined = docs + augmented
    print(f"\nWriting {len(combined)} documents to {OUTPUT}")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for doc in combined:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()
