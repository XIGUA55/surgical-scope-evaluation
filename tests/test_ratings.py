import numpy as np

from scope_eval.ratings import icc_2_1


def test_icc_is_one_for_identical_raters():
    ratings = np.array([[1, 1], [2, 2], [4, 4], [5, 5]], dtype=float)
    assert np.isclose(icc_2_1(ratings), 1.0)
