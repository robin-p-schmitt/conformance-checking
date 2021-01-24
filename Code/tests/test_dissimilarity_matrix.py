import numpy as np
import os
import pytest
import conformance_checking as cf


def test_calc_fitness_precision_1():
    matrix = cf.DissimilarityMatrix(np.asarray([
        # dissimilarities are maximal for all model/real trace pairs
        # thus, precision and fitness is minimal
        [1.0, 1.0],
        [1.0, 1.0],
    ]))
    assert matrix.calc_precision() == pytest.approx(0.0)
    assert matrix.calc_fitness() == pytest.approx(0.0)


def test_calc_fitness_precision_2():
    matrix = cf.DissimilarityMatrix(np.asarray([
        # dissimilarities are minimal for all model/real trace pairs
        # thus, precision and fitness is maximal (perfect match)
        [0.0, 0.0],
        [0.0, 0.0],
    ]))
    assert matrix.calc_precision() == pytest.approx(1.0)
    assert matrix.calc_fitness() == pytest.approx(1.0)


def test_calc_fitness_precision_3():
    matrix = cf.DissimilarityMatrix(np.asarray([
        # each real trace (column) can be represented by the model (traces in rows),
        # but the model allows for extra traces not covered by the real log
        # thus, fitness is maximal but precision is low
        [0.0, 0.0],
        [1.0, 1.0],
    ]))
    assert matrix.calc_precision() == pytest.approx(0.5)
    assert matrix.calc_fitness() == pytest.approx(1.0)


def test_calc_fitness_precision_4():
    matrix = cf.DissimilarityMatrix(np.asarray([
        # each model trace (row) is equal to a real trace (column)
        # but the model does not cover all real traces
        # thus, precision is maximal but fitness is low
        [1.0, 0.0],
        [1.0, 0.0],
    ]))
    assert matrix.calc_precision() == pytest.approx(1.0)
    assert matrix.calc_fitness() == pytest.approx(0.5)


def test_calc_fitness_precision_5():
    # this example just checks for floating point semantics
    # all dissimilarities are high, thus precision and fitness is low
    matrix = cf.DissimilarityMatrix(np.asarray([
        [1 - 1e-9, 1 - 1e-9],
        [1 - 1e-9, 1 - 1e-9],
    ]))
    assert matrix.calc_precision() == pytest.approx(1e-9)
    assert matrix.calc_fitness() == pytest.approx(1e-9)


def test_save_load(tmpdir):
    path = os.path.join(tmpdir, "dissimilarity_matrix.npy")
    matrix = np.asarray([[1, 0], [0.5, 0.25]], dtype=np.float32)
    reference = cf.DissimilarityMatrix(matrix)
    reference.save(path)
    loaded = cf.DissimilarityMatrix.load(path)
    assert np.all(matrix == loaded.get_dissimilarity_matrix())
