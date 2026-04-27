from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import hashlib
import os, re

import edsnlp


@dataclass(frozen=True)
class DetectedSpan:
    """A single detected PHI span to replace."""
    start: int
    end: int
    label: str
    text: str
    pseudo_value: str


@dataclass(frozen=True)
class PractitionerSpan:
    """A detected practitioner name span (preceded by a medical title)."""
    name: str
    title: str
    start: int
    end: int


# Matches "Dr Touat", "Pr. SANSON", "Professeur Hoang-Xuan", "du Professeur Bielle",
# "Docteur Laigle-Donadey", "Dr TOUAT MEHDI", etc.
# Handles: ALL CAPS names, OCR typos in titles, optional prepositions.
_TITLE_NAME_PATTERN = re.compile(
    r"(?:(?:du|le|par\s+le|par)\s+)?"                          # optional preposition
    r"(?P<title>"
    r"Dr\.?"                                                    # Dr / Dr.
    r"|Pr\.?"                                                   # Pr / Pr.
    r"|Prof(?:esseure?|esscure?)?"                              # Professeur(e) + OCR variant
    r"|Doct(?:eure?|eur)?"                                      # Docteur(e) + OCR variant
    r"|Interne"
    r")"
    r"\s+"
    r"(?P<name>"
    r"[A-ZÀ-ÜÉ][A-Za-zÀ-ÖØ-öø-ÿ\u0100-\u017F'-]*\.?"            # first word (Title, UPPER, or Initial with optional dot)
    r"(?:[\s-][A-ZÀ-ÜÉ][A-Za-zÀ-ÖØ-öø-ÿ\u0100-\u017F'-]*\.?)*"  # subsequent hyphenated/spaced words
    r")",
    re.UNICODE | re.IGNORECASE,
)

# Known BRIGHT team and affiliated practitioners at Pitié-Salpêtrière
BRIGHT_PRACTITIONERS: set[str] = {
    # Neuro-oncologists
    "Hoang-Xuan", "Touat", "Sanson", "Idbaih", "Alentorn",
    "Laigle-Donadey", "Donadey", "Leprince-Laurenge",
    "Dehais", "Picca", "Psimaras", "Houillier", "Louis",
    "Ferrari", "Delattre", "Bielle",
    # Neurosurgeons
    "Mathon", "Marijon", "Nouet", "Carpentier", "Peyre", "Lefevre",
    # Radiation therapists
    "Assouline", "Maingon", "Michel",
}

# Pre-compute lowercased whitelist for case-insensitive matching
_BRIGHT_PRACTITIONERS_LOWER: set[str] = {n.lower() for n in BRIGHT_PRACTITIONERS}


class TextPseudonymizer:
    """
    Wraps an EDSNLP (eds-pseudo) pipeline and rewrites text based on detected entities.

    Key properties:
    - Replacement is done from right-to-left to preserve offsets.
    - Pseudonyms are deterministic per (ipp, label, original_text) using a secret salt,
      so repeated occurrences map to the same token across documents if desired.
    """

    def __init__(
        self,
        model_path: str,
        *,
        hf_cache_dir: Optional[str] = None,
        secret_salt: Optional[str] = None,
        auto_update: bool = False,
        keep: list[str] = ["IPP", "NDA", "DATE", "HOPITAL"]
    ) -> None:
        """
        model_path: local path to eds-pseudo artifacts directory (the path you pass to edsnlp.load)
        hf_cache_dir: optional, if you want to manage HF downloads elsewhere (not required if already local)
        secret_salt: used to make deterministic pseudonyms non-reversible; default uses env var PSEUDO_SALT
        auto_update: keep False if you manage downloads yourself (as in your test_eds.py)
        """
        self.model_path = model_path
        self.hf_cache_dir = hf_cache_dir
        self.secret_salt = secret_salt or os.environ.get("PSEUDO_SALT", "CHANGE_ME")
        self.nlp = edsnlp.load(model_path, auto_update=auto_update)
        self.keep = keep

    # ---------------------------
    # Public API
    # ---------------------------

    def detect_practitioner_names(self, text: str) -> List[PractitionerSpan]:
        """Detect practitioner names preceded by medical title prefixes.

        Returns a list of PractitionerSpan objects for names found after
        Dr, Pr, Professeur, Docteur, or Interne.
        """
        results: List[PractitionerSpan] = []
        for m in _TITLE_NAME_PATTERN.finditer(text):
            results.append(PractitionerSpan(
                name=m.group("name"),
                title=m.group("title"),
                start=m.start(),
                end=m.end(),
            ))
        return results

    def pseudonymize(
        self,
        text: str,
        *,
        ipp: Optional[str] = None,
        consistent_across_ipp: bool = False,
        label_to_template: Optional[Dict[str, str]] = None,
        keep_practitioner_names: bool = True,
    ) -> str:
        """
        Returns a pseudonymized version of the provided text.

        ipp:
          If provided, can be used to generate per-patient deterministic pseudonyms.
        consistent_across_ipp:
          - False (recommended): pseudonyms are stable within the patient (ipp) but different across patients.
          - True: pseudonyms are stable globally across all patients (useful for longitudinal cross-patient linking;
                  usually not desired for strict de-identification).
        label_to_template:
          Optional override map. Template receives {label} and {token}.
          Example: {"PERSON": "[PERSON_{token}]"}
        """
        spans = self.detect_spans(text)
        if not spans:
            return text

        # Default templates: readable, explicit placeholders.
        # You should tune these labels to the ones produced by your eds-pseudo model.

        templates = {
            "NOM": "[NOM_{token}]",
            "PRENOM": "[PRENOM_{token}]",
            "TEL": "[TEL_{token}]",
            "MAIL": "[MAIL_{token}]",
            "DATE": "[DATE_{token}]",
            "DATE_NAISSANCE": "{pseudo_value}",
            "ADRESSE": "[ADDRESS_{token}]",
            "ZIP": "[ZIP_{token}]",
            "VILLE": "[VILLE_{token}]",
            # "HOPITAL": "[HOPITAL_{token}]", we want to identify localization
            "IPP": "[IPP_{token}]",
            "NDA": "[NDA_{token}]",
            "SECU": "[SSID_{token}]"
        }
        if label_to_template:
            templates.update(label_to_template)

        # Build set of practitioner name ranges for preservation
        practitioner_ranges: List[Tuple[int, int]] = []
        if keep_practitioner_names:
            for ps in self.detect_practitioner_names(text):
                practitioner_ranges.append((ps.start, ps.end))

        # Build replacements
        replacements: List[Tuple[int, int, str]] = []
        for sp in spans:
            if sp.label not in self.keep:
                # Skip pseudonymisation of practitioner names
                if keep_practitioner_names and sp.label in ("NOM", "PRENOM"):
                    # Check title-based detection: span overlaps with a practitioner span
                    is_practitioner = any(
                        ps_start <= sp.start and sp.end <= ps_end
                        for ps_start, ps_end in practitioner_ranges
                    )
                    # Check whitelist fallback
                    if not is_practitioner:
                        is_practitioner = sp.text.strip().lower() in _BRIGHT_PRACTITIONERS_LOWER
                    if is_practitioner:
                        continue  # Preserve this name
                template = templates.get(sp.label, "[PHI_{label}_{token}]")
                token = self._stable_token(
                    sp.text,
                    label=sp.label,
                    ipp=ipp,
                    consistent_across_ipp=consistent_across_ipp,
                )
                if sp.label == "DATE_NAISSANCE":
                    repl = template.format(pseudo_value=sp.pseudo_value)
                else:
                    repl = template.format(label=sp.label, token=token)
                replacements.append((sp.start, sp.end, repl))

        # Apply right-to-left to avoid offset shift
        return self._apply_replacements(text, replacements)

    def detect_spans(self, text: str) -> List[DetectedSpan]:
        """
        Runs the model and extracts entity spans.

        Chunking strategy:
          - Split large inputs into overlapping chunks to stay under the model max length.
          - Run the pipeline per chunk.
          - Recombine spans by offsetting start/end with the chunk start.
          - Dedupe overlaps globally (keeps the longest span in each overlap region).
        """
        MAX_MODEL_CHARS = 1000
        OVERLAP_CHARS = 350

        all_spans: List[DetectedSpan] = []

        for chunk, offset in self._iter_text_chunks(
            text,
            max_chars=MAX_MODEL_CHARS,
            overlap=OVERLAP_CHARS,
        ):
            if not chunk.strip():
                continue

            doc = self.nlp(chunk)
            for ent in getattr(doc, "ents", []):
                start = int(ent.start_char) + offset
                end = int(ent.end_char) + offset
                if start < 0 or end <= start or end > len(text):
                    continue
                
                if str(ent.label_) == "DATE_NAISSANCE":
                    pseudo_value = str(ent._.date).split("-")[0] + "-??-??"
                    span = DetectedSpan(
                        start=start,
                        end=end,
                        label=str(ent.label_),
                        text=text[start:end],
                        pseudo_value=pseudo_value,
                    )
                else:
                    span = DetectedSpan(
                        start=start,
                        end=end,
                        label=str(ent.label_),
                        text=text[start:end],
                        pseudo_value="",
                    )

                all_spans.append(span)

        if not all_spans:
            return []

        # Sort by (start asc, end desc) for predictable behavior, then drop overlaps.
        all_spans.sort(key=lambda s: (s.start, -(s.end - s.start)))
        all_spans = self._dedupe_overlaps(all_spans)
        return all_spans

    # ---------------------------
    # Internals
    # ---------------------------

    def _stable_token(
        self,
        original: str,
        *,
        label: str,
        ipp: Optional[str],
        consistent_across_ipp: bool,
    ) -> str:
        """
        Deterministic short token derived from secret salt + (ipp scope) + label + original text.
        """
        scope = "GLOBAL" if consistent_across_ipp else (ipp or "NO_ipp")
        payload = f"{self.secret_salt}|{scope}|{label}|{original}".encode("utf-8", errors="ignore")
        digest = hashlib.sha256(payload).hexdigest()
        return digest[:10].upper()

    @staticmethod
    def _apply_replacements(text: str, replacements: Iterable[Tuple[int, int, str]]) -> str:
        """
        replacements: iterable of (start, end, replacement_string)
        Applied from right-to-left.
        """
        reps = sorted(replacements, key=lambda x: x[0], reverse=True)
        out = text
        for start, end, repl in reps:
            if start < 0 or end > len(out) or start >= end:
                continue
            out = out[:start] + repl + out[end:]
        return out

    @staticmethod
    def _dedupe_overlaps(spans: List[DetectedSpan]) -> List[DetectedSpan]:
        """
        Remove overlapping spans by keeping the longest span in each overlap region.
        Assumes spans are sorted by (start asc, length desc).
        """
        kept: List[DetectedSpan] = []
        current_end = -1
        for sp in spans:
            if sp.start >= current_end:
                kept.append(sp)
                current_end = sp.end
            else:
                # overlap: skip because we kept an earlier (and likely longer) one
                # given the sorting (start asc, end desc)
                continue
        return kept
    
    @staticmethod
    def _find_safe_split_point(text: str, start: int, hard_end: int, *, min_end: int) -> int:
        """
        Choose a split point <= hard_end, preferring natural boundaries:
          1) blank line
          2) newline
          3) sentence terminators (.!?)
          4) weaker punctuation (;:)
          5) whitespace
        Also ensures we never cut in the middle of a word.
        Returns an absolute index in [min_end, hard_end] (or hard_end if forced).
        """
        if hard_end <= min_end:
            end = hard_end
        else:
            window = text[min_end:hard_end]

            # Order matters: strongest boundaries first.
            patterns = (
                r"\n\s*\n",          # paragraph break
                r"\n",               # newline
                r"[\.!?](?:\s+|$)",  # sentence end
                r"[;:](?:\s+|$)",    # weaker sentence-like break
                r"\s+",              # any whitespace
            )

            end = None
            for pat in patterns:
                matches = list(re.finditer(pat, window))
                if matches:
                    m = matches[-1]
                    end = min_end + m.end()
                    break

            if end is None:
                end = hard_end

        # Word-boundary safety: if we're in the middle of an alnum "word", back off to a safe boundary.
        # Example: "...JohnDoe|Smith..." -> move 'end' left to whitespace/punct.
        if 0 < end < len(text):
            left = text[end - 1]
            right = text[end]
            if left.isalnum() and right.isalnum():
                j = end
                # Back off until we hit a non-alnum boundary or reach start/min_end.
                while j > min_end and text[j - 1].isalnum():
                    j -= 1
                # If backing off moved too far (rare), fall back to hard_end.
                if j >= min_end:
                    end = j

        # Final clamp
        end = max(min_end, min(end, hard_end))
        return end

    @classmethod
    def _iter_text_chunks(
        cls,
        text: str,
        *,
        max_chars: int,
        overlap: int,
        min_chunk_chars: int = 200,
    ):
        """
        Yield (chunk_text, chunk_start_offset) with overlap to catch entities crossing boundaries.
        """
        n = len(text)
        if n <= max_chars:
            yield text, 0
            return

        if overlap < 0:
            overlap = 0
        overlap = min(overlap, max_chars // 3)

        start = 0
        while start < n:
            hard_end = min(start + max_chars, n)
            min_end = min(start + max(min_chunk_chars, max_chars - 2 * overlap), hard_end)

            end = cls._find_safe_split_point(text, start, hard_end, min_end=min_end)

            chunk = text[start:end]
            yield chunk, start

            if end >= n:
                break

            # Move forward but keep overlap
            start = max(0, end - overlap)
            if start >= n:
                break


if __name__ == "__main__":
    from pathlib import Path
    from utils import prepare_eds_registry
    import os, sys, textwrap

    os.chdir(Path(__file__).resolve().parent)

    artifacts_path = Path("../../hf_cache/artifacts").resolve()
    prepare_eds_registry(artifacts_path.parent)
    TP = TextPseudonymizer(str(artifacts_path))

    # ──────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────
    _pass_count = 0
    _fail_count = 0

    def check(condition: bool, description: str) -> None:
        global _pass_count, _fail_count
        if condition:
            _pass_count += 1
            print(f"  ✅ PASS, {description}")
        else:
            _fail_count += 1
            print(f"  ❌ FAIL, {description}")

    def section(title: str) -> None:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")

    # ──────────────────────────────────────────────────────────
    # Test text, realistic clinical report mixing patient PHI
    # and practitioner names in many formats.
    # ──────────────────────────────────────────────────────────
    test_text = textwrap.dedent("""\
        Références : ALE/ALE
        Compte-Rendu de Consultation du 01/12/2025

        Madame LAURENGE ep. LEPRINCE Alice, née le 18/05/1989, âgée de 36 ans,
        a été vue en consultation par le Dr Touat dans le service du Pr Sanson à l'hopital Pitié Salpétriere.
        La patiente habite au 12 rue des Lilas, 75013 Paris.
        Téléphone : 06 12 34 56 78.  E-mail : alice.laurenge@mail.com

        Avis du Professeur Hoang-Xuan, du Pr. SANSON et du Docteur Laigle-Donadey.
        Interne Picca a également participé à la consultation.

        Dr TOUAT MEHDI a prescrit une IRM cérébrale.
        Professeure Bielle a relu la lame anatomo-pathologique.

        Pat.: Alice LAURENGE (Nom usuel : LEPRINCE) | F | 18/05/1989
        INS/NIR : 289055951218119 | IPP 8008897828 | NDA 6602777525

        Imprimé le 01/12/2025 17:22
        CR CONSULTATION PSL NEURO-ONCOLOGIE
    """)

    result = TP.pseudonymize(test_text, ipp="TEST_IPP", keep_practitioner_names=True)

    # ──────────────────────────────────────────────────────────
    # 1. Patient PHI must be pseudonymized
    # ──────────────────────────────────────────────────────────
    section("1 · Patient PHI is pseudonymized")

    # Patient names, should NOT appear verbatim
    check("LAURENGE" not in result,
          "Patient surname LAURENGE is replaced")
    check("LEPRINCE" not in result,
          "Patient maiden name LEPRINCE is replaced")
    check("Alice" not in result,
          "Patient first name Alice is replaced")

    # Pseudonymization placeholders should be present
    check("[NOM_" in result or "[PRENOM_" in result,
          "At least one name placeholder token is present")

    # ──────────────────────────────────────────────────────────
    # 2. Practitioner names must be PRESERVED
    # ──────────────────────────────────────────────────────────
    section("2 · Practitioner names are preserved")

    # Title-prefixed names (regex-detected)
    check("Touat" in result,
          "Dr Touat is preserved (title-case after Dr)")
    check("Sanson" in result or "SANSON" in result,
          "Pr Sanson / SANSON is preserved")
    check("Hoang-Xuan" in result,
          "Professeur Hoang-Xuan is preserved (hyphenated)")
    check("Laigle-Donadey" in result,
          "Docteur Laigle-Donadey is preserved (hyphenated)")
    check("Picca" in result,
          "Interne Picca is preserved")
    check("Bielle" in result,
          "Professeure Bielle is preserved")
    check("TOUAT" in result or "Touat" in result,
          "Dr TOUAT MEHDI (ALL CAPS after Dr) is preserved")

    # ──────────────────────────────────────────────────────────
    # 3. detect_practitioner_names regex unit tests
    # ──────────────────────────────────────────────────────────
    section("3 · detect_practitioner_names regex")

    regex_cases = [
        ("Dr Touat",                   "Touat",          "Dr + Title-case"),
        ("Pr. SANSON",                 "SANSON",         "Pr. + ALL CAPS"),
        ("Professeur Hoang-Xuan",      "Hoang-Xuan",    "Professeur + hyphenated"),
        ("du Professeur Bielle",       "Bielle",         "du Professeur"),
        ("Docteur Laigle-Donadey",     "Laigle-Donadey", "Docteur + hyphenated"),
        ("Dr TOUAT MEHDI",             "TOUAT MEHDI",    "Dr + two ALL-CAPS words"),
        ("Interne Picca",              "Picca",          "Interne + name"),
        ("par le Pr Idbaih",           "Idbaih",         "par le Pr"),
        ("Professeure Bielle",         "Bielle",         "Professeure (feminine)"),
        ("Prof Delattre",              "Delattre",       "Prof (abbreviated)"),
        ("Docteure Houillier",         "Houillier",      "Docteure (feminine)"),
    ]
    for snippet, expected_name, desc in regex_cases:
        spans = TP.detect_practitioner_names(snippet)
        names = [s.name for s in spans]
        check(expected_name in names,
              f"Regex detects '{expected_name}' in \"{snippet}\" ({desc})")

    # Negative case: bare patient name without title must NOT match
    neg_spans = TP.detect_practitioner_names("LAURENGE Alice")
    check(len(neg_spans) == 0,
          "Bare patient name without title is NOT detected as practitioner")

    # ──────────────────────────────────────────────────────────
    # 4. Whitelist fallback: names in BRIGHT_PRACTITIONERS
    #    even without a title prefix should survive
    # ──────────────────────────────────────────────────────────
    section("4 · Whitelist-only preservation (no title prefix)")

    whitelist_text = "Consultation avec Sanson et Ferrari pour avis complémentaire."
    wl_result = TP.pseudonymize(whitelist_text, ipp="TEST_WL", keep_practitioner_names=True)
    # If the NLP model detects "Sanson" / "Ferrari" as NOM/PRENOM, the whitelist
    # should prevent them from being replaced.
    # We check that the names still appear OR that no NOM placeholder replaced them.
    check("Sanson" in wl_result or "[NOM_" not in wl_result,
          "Whitelist preserves 'Sanson' when it would be detected as NOM")
    check("Ferrari" in wl_result or "[NOM_" not in wl_result,
          "Whitelist preserves 'Ferrari' when it would be detected as NOM")

    # ──────────────────────────────────────────────────────────
    # 5. keep_practitioner_names=False ⇒ practitioners ARE pseudonymized
    # ──────────────────────────────────────────────────────────
    section("5 · keep_practitioner_names=False replaces everything")

    result_no_keep = TP.pseudonymize(test_text, ipp="TEST_IPP", keep_practitioner_names=False)
    # At least one practitioner name should now be gone (replaced by placeholder)
    all_present = all(name in result_no_keep for name in ["Touat", "Sanson", "Hoang-Xuan", "Bielle"])
    check(not all_present,
          "With keep_practitioner_names=False, at least some practitioner names are pseudonymized")

    # ──────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────
    section("RESULTS")
    total = _pass_count + _fail_count
    print(f"\n  {_pass_count}/{total} passed, {_fail_count} failed.\n")

    print("─" * 60)
    print("  Pseudonymized output (with practitioners preserved):")
    print("─" * 60)
    print(result)

    if _fail_count > 0:
        sys.exit(1)
    else:
        print("\n🎉 All tests passed!\n")
        sys.exit(0)