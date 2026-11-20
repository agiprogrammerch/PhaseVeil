"""Root pytest configuration - lets the test suite run with **zero installation**.

With this file present, ``python3 -m pytest`` and ``pytest`` (run from the
repository root) import the package straight from ``src/`` without needing
``pip install -e .``, a virtual environment, or ``PYTHONPATH``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))