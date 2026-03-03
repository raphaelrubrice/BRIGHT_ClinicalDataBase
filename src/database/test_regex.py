import re

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
    r"(?:[A-ZÀ-ÜÉ](?:\.|[A-Za-zÀ-ÖØ-öø-ÿ\u0100-\u017F'-]*))" # first word or initial
    r"(?:[\s-][A-ZÀ-ÜÉ](?:\.|[A-Za-zÀ-ÖØ-öø-ÿ\u0100-\u017F'-]*))*"  # subsequent
    r")",
    re.UNICODE | re.IGNORECASE,
)

print(_TITLE_NAME_PATTERN.search('Docteur A. A. M. LEPRINCE\nLAURENGE'))
print(_TITLE_NAME_PATTERN.search('Dr TOUAT MEHDI'))
print(_TITLE_NAME_PATTERN.search('Professeure Bielle'))
