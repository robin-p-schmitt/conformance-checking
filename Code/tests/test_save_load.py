import numpy as np
import os
import conformance_checking as cf


def tests_save_load(tmpdir):
    path = os.path.join(tmpdir, "dissimilarity_matrix.npy")
    matrix = np.asarray([[1, 0], [0.5, 0.25]], dtype=np.float32)
    reference = cf.DissimilarityMatrix(matrix)
    reference.save(path)
    loaded = cf.DissimilarityMatrix.load(path)
    assert np.all(matrix == loaded.get_dissimilarity_matrix())
