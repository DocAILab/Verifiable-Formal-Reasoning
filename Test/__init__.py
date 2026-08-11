"""Evaluation package using the exact training-side formal logic stack."""

from __future__ import annotations

import sys
from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parents[1] / "Training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

