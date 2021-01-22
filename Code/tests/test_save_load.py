import numpy as np
import os


def tests_save_load(tmpdir):
    path = os.path.join(tmpdir, "dissimilarity_matrix.npy")
    matrix = np.asarray([[1, 0], [0.5, 0.25]], dtype=np.float32)
    reference = DissimilarityMatrix(matrix)
    reference.save(path)
    loaded = DissimilarityMatrix.load(path)
    assert np.all(matrix == loaded.get_dissimilarity_matrix())
