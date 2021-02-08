import pytest
import numpy as np

from conformance_checking.distances import calc_wmd, calc_ict, calc_euclidean, _calc_d


def test_wmd():
    """Is the wmd calculated correctly?"""
    model_embedding = [{0: 1, 1: 1}]
    real_embedding = [{1: 1}]
    context = np.array([[1, 4], [5, 1]])

    # calculate Euclidean distance matrix
    distance_matrix = calc_euclidean(context)

    # calc d for embeddings
    vocab_len = len(context)
    d_model = _calc_d(model_embedding, vocab_len)
    d_real = _calc_d(real_embedding, vocab_len)

    assert calc_wmd(d_model[0], d_real[0], distance_matrix) == pytest.approx(2.5)


def test_ict():
    """Is the ict calculated correctly?"""
    model_embedding = [{0: 1, 1: 1}]
    real_embedding = [{1: 1}]
    context = np.array([[1, 4], [5, 1]])

    # calculate Euclidean distance matrix
    distance_matrix = calc_euclidean(context)

    # calc d for embeddings
    vocab_len = len(context)
    d_model = _calc_d(model_embedding, vocab_len)
    d_real = _calc_d(real_embedding, vocab_len)

    assert calc_ict(d_model[0], d_real[0], distance_matrix) == pytest.approx(2.5)


def test_ict2():
    """Is the ict calculated correctly?"""
    model_embedding = [{0: 1}]
    real_embedding = [{1: 1}]
    context = np.array([[0, 4], [500, 4]])

    # calculate Euclidean distance matrix
    distance_matrix = calc_euclidean(context)

    # calc d for embeddings
    vocab_len = len(context)
    d_model = _calc_d(model_embedding, vocab_len)
    d_real = _calc_d(real_embedding, vocab_len)

    assert calc_ict(d_model[0], d_real[0], distance_matrix) == pytest.approx(500)
