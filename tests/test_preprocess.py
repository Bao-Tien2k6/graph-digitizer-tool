from pathlib import Path

import numpy as np
import pytest

from pipeline.preprocess import preprocess_image

SAMPLE_DIR = Path(__file__).parent / "sample_figures"


def _first_sample() -> Path:
    """Return any sample figure, or skip if none are checked in."""
    samples = sorted(SAMPLE_DIR.rglob("*.png"))
    if not samples:
        pytest.skip(f"No sample figures found under {SAMPLE_DIR}")
    return samples[0]


def test_preprocess_with_sample():
    result = preprocess_image(str(_first_sample()))
    assert isinstance(result, np.ndarray)
    assert result.ndim == 3
    assert result.shape[2] == 3  # BGR image
    assert result.dtype == np.uint8
