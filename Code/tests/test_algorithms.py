from conformance_checking.algorithms import (
    Act2VecWmdConformance,
    Act2VecIctConformance,
    Trace2VecCosineConformance,
)
from tests.test_algorithm_baseclass import check_algorithm

import numpy as np
import pytest


def example_log_a():
    return [
        ["hi", "foo"],
        ["hi", "foo"],
        ["bar"],
        [],
        ["a", "long", "trace", "with", "doubled", "words", "like", "long"],
    ]


def example_log_b():
    return [
        ["foobar", "hi"],
        ["bar"],
        ["bar"],
        [],
        ["a", "long", "long", "trace", "but", "not", "the", "same"],
    ]


def test_act2vec_wmd_check():
    check_algorithm(Act2VecWmdConformance())


def test_act2vec_wmd_equal():
    dissimilarity_matrix = Act2VecWmdConformance().execute(
        example_log_a(), example_log_a()
    )
    np_matrix = dissimilarity_matrix.get_dissimilarity_matrix()
    n = np_matrix.shape[0]
    assert np.sum(np_matrix * np.eye(n)) == pytest.approx(0, abs=1e-6)


def test_act2vec_wmd_not_equal():
    dissimilarity_matrix = Act2VecWmdConformance().execute(
        example_log_a(), example_log_b()
    )
    np_matrix = dissimilarity_matrix.get_dissimilarity_matrix()
    assert np.any(np_matrix != 0.0)


def test_act2vec_ict_check():
    check_algorithm(Act2VecIctConformance())


def test_act2vec_ict_equal():
    dissimilarity_matrix = Act2VecIctConformance().execute(
        example_log_a(), example_log_a()
    )
    np_matrix = dissimilarity_matrix.get_dissimilarity_matrix()
    n = np_matrix.shape[0]
    assert np.sum(np_matrix * np.eye(n)) == pytest.approx(0, abs=1e-6)


def test_act2vec_ict_not_equal():
    dissimilarity_matrix = Act2VecIctConformance().execute(
        example_log_a(), example_log_a()
    )
    np_matrix = dissimilarity_matrix.get_dissimilarity_matrix()
    assert np.any(np_matrix != 0.0)


def test_trace2vec_cosine_check():
    check_algorithm(Trace2VecCosineConformance())


def test_trace2vec_cosine_equal():
    dissimilarity_matrix = Trace2VecCosineConformance().execute(
        example_log_a(), example_log_a()
    )
    np_matrix = dissimilarity_matrix.get_dissimilarity_matrix()
    n = np_matrix.shape[0]
    assert np.sum(np_matrix * np.eye(n)) == pytest.approx(0, abs=1e-6)


def test_trace2vec_cosine_not_equal():
    dissimilarity_matrix = Trace2VecCosineConformance().execute(
        example_log_a(), example_log_a()
    )
    np_matrix = dissimilarity_matrix.get_dissimilarity_matrix()
    assert np.any(np_matrix != 0.0)
