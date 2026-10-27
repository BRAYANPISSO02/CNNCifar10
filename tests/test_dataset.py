import numpy as np
from src.data.dataset import filter_classes


def test_filter_classes():
    X = np.random.randn(10, 32, 32, 3)
    y = np.array([0, 1, 2, 3, 4, 0, 1, 2, 8, 9])
    target_classes = [0, 1, 2]

    X_filt, y_filt = filter_classes(X, y, target_classes)

    assert X_filt.shape[0] == 6
    assert len(y_filt) == 6
    assert set(y_filt).issubset({0, 1, 2})
    assert np.array_equal(y_filt, np.array([0, 1, 2, 0, 1, 2]))
