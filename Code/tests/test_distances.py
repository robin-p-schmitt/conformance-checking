import pytest
import numpy as np

from conformance_checking.distances import calc_wmd, calc_ict


def test_wmd():
    """Is the wmd calculated correctly?"""
    model_embedding = {0: 1, 1: 1}
    real_embedding = {1: 1}
    context = np.array([[1, 4], [5, 1]])
    assert calc_wmd(model_embedding, real_embedding, context) == pytest.approx(2.5)


def test_ict():
    """Is the ict calculated correctly?"""
    model_embedding = {0: 1, 1: 1}
    real_embedding = {1: 1}
    context = np.array([[1, 4], [5, 1]])
    assert calc_ict(model_embedding, real_embedding, context) == pytest.approx(2.5)


def test_ict2():
    """Is the ict calculated correctly?"""
    model_embedding = {0: 1}
    real_embedding = {1: 1}
    context = np.array([[0, 4], [500, 4]])
    assert calc_ict(model_embedding, real_embedding, context) == pytest.approx(500)
