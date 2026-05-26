from pipeline.preprocess import preprocess_image
import numpy as np


def test_preprocess_with_sample():
    result = preprocess_image(r"tests\sample_figures\line_1.png")
    assert isinstance(result, np.ndarray)
    assert result.shape[2] == 3  # BGR image
    assert result.dtype == np.uint8

test_preprocess_with_sample()
