"""Root conftest, ensure the project root is on sys.path so that
``from src.extraction import …`` works when running pytest from the
BRIGHT_ClinicalDataBase directory.
"""

import sys
from pathlib import Path

# Add the project root (where 'src/' lives) to sys.path
_root = str(Path(__file__).resolve().parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
