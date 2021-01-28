import numpy as np
import pytest
from conformance_checking import EmbeddingConformance


def example():
    class Mock(EmbeddingConformance):
        def _calc_embeddings(self, model_traces, real_traces):
            model_embeddings = np.asarray(
                [len(t) for t in model_traces], dtype=np.float32
            )
            real_embeddings = np.asarray(
                [len(t) for t in real_traces], dtype=np.float32
            )
            return model_embeddings, real_embeddings, 1

        def _calc_dissimilarity(self, model_embedding, real_embedding, context):
            assert context == 1
            max_len = max(model_embedding, real_embedding)
            min_len = min(model_embedding, real_embedding)
            if max_len == 0:
                return 0
            else:
                return 1 - min_len / max_len

    model_traces = [
        ["hi", "foo"],
        ["hi", "foo"],
        ["bar"],
        [],
        ["a", "long", "trace", "with", "doubled", "words", "like", "long"],
    ]
    real_traces = [
        ["foobar", "hi"],
        ["bar"],
        ["bar"],
        [],
        ["a", "long", "long", "trace", "but", "not", "the", "same"],
    ]
    expected_matrix = np.asarray(
        [
            [0, 0.5, 0.5, 1, 0.75],
            [0, 0.5, 0.5, 1, 0.75],
            [0.5, 0, 0, 1, 0.875],
            [1, 1, 1, 0, 1],
            [0.75, 0.875, 0.875, 1, 0],
        ],
        dtype=np.float32,
    )
    return Mock(), model_traces, real_traces, expected_matrix


def check_algorithm(algorithm):
    _, model_traces, real_traces, _ = example()

    # check embeddings
    model_embeddings, real_embeddings, context = algorithm._calc_embeddings(
        model_traces, real_traces
    )
    assert len(model_embeddings) == len(
        model_traces
    ), "There must be as many model embeddings as model traces!"
    assert len(real_embeddings) == len(
        real_traces
    ), "There must be as many real embeddings as real traces!"

    # check dissimilarity function
    for model_trace, model_embedding in zip(model_traces, model_embeddings):
        for real_trace, real_embedding in zip(model_traces, model_embeddings):
            dissimilarity = algorithm._calc_dissimilarity(
                model_embedding, real_embedding, context
            )
            assert 0 <= dissimilarity <= 1, "Dissimilarity values should be in [0,1]!"
            if model_trace == real_trace:
                assert dissimilarity == pytest.approx(
                    0, abs=1e-6
                ), "Equal traces should have a dissimilarity of zero!"


def test_check_algorithm():
    algorithm, _, _, _ = example()
    check_algorithm(algorithm)


def test_algorithm_execution():
    algorithm, model_traces, real_traces, expected_matrix = example()
    result = algorithm.execute(model_traces, real_traces)
    matrix = result.get_dissimilarity_matrix()
    assert isinstance(
        matrix, np.ndarray
    ), "Dissimilarity matrix should be a numpy array!"
    assert matrix.dtype == np.float32, "Dissimilarity matrix should be of type float32!"
    assert np.all(matrix == expected_matrix), "Expected:\n%s\nGot:\n%s" % (
        str(expected_matrix),
        str(matrix),
    )
