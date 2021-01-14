import pytest

from conformance_checking.distances import calc_wmd, calc_ict


def test_wmd():
    """Is the wmd calculated correctly?"""
    model_embedding = {(1, 4): 1, (5, 1): 1}
    real_embedding = {(5, 1): 1}
    assert calc_wmd(model_embedding, real_embedding) == pytest.approx(2.5)


def test_ict():
    """Is the ict calculated correctly?"""
    model_embedding = {(1, 4): 1, (5, 1): 1}
    real_embedding = {(5, 1): 1}
    assert calc_ict(model_embedding, real_embedding) == pytest.approx(2.5)


def test_ict2():
    """Is the ict calculated correctly?"""
    model_embedding = {(0, 4): 1}
    real_embedding = {(500, 4): 1}
    assert calc_ict(model_embedding, real_embedding) == pytest.approx(500)
