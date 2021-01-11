import numpy as np
from conformance_checking import EmbeddingConformance


def example():
    class Mock(EmbeddingConformance):
        @staticmethod
        def _calc_embeddings(model_traces, real_traces):
            model_embeddings = np.asarray(
                [[len(t)] for t in model_traces], dtype=np.float32
            )
            real_embeddings = np.asarray(
                [[len(t)] for t in real_traces], dtype=np.float32
            )
            return model_embeddings, real_embeddings

        @staticmethod
        def _calc_dissimilarity(model_embedding, real_embedding):
            max_len = max(model_embedding[0], real_embedding[0])
            min_len = min(model_embedding[0], real_embedding[0])
            if max_len == 0:
                return 0
            else:
                return 1 - min_len / max_len

    model_traces = [
        ["hi", "foo"],
        ["bar"],
        [],
        ["a", "long", "trace", "with", "doubled", "words", "like", "long"],
    ]
    real_traces = [
        ["foobar", "hi"],
        ["bar"],
        [],
        ["a", "long", "long", "trace", "but", "not", "the", "same"],
    ]
    expected_matrix = np.asarray(
        [[0, 0.5, 1, 0.75], [0.5, 0, 1, 0.875], [1, 1, 0, 1], [0.75, 0.875, 1, 0]],
        dtype=np.float32,
    )
    return Mock, model_traces, real_traces, expected_matrix


def check_algorithm(algorithm):
    _, model_traces, real_traces, _ = example()

    # check inheritance
    assert issubclass(
        algorithm, EmbeddingConformance
    ), "Algorithm must subclass EmbeddingConformance!"

    # check embeddings
    model_embeddings, real_embeddings = algorithm._calc_embeddings(
        model_traces, real_traces
    )
    assert isinstance(
        model_embeddings, np.ndarray
    ), "Model embeddings should be a NumPy array!"
    assert isinstance(
        real_embeddings, np.ndarray
    ), "Real embeddings should be a NumPy array!"
    assert (
        model_embeddings.dtype == np.float32
    ), "Model embeddings should have type float32!"
    assert (
        real_embeddings.dtype == np.float32
    ), "Real embeddings should have type float32!"
    assert len(model_embeddings.shape) >= 2 and model_embeddings.shape[0] == len(
        model_traces
    ), (
        "Model embeddings should be of shape num_traces x embedding_shape! Got: %s"
        % str(model_embeddings.shape)
    )
    assert len(real_embeddings.shape) >= 2 and real_embeddings.shape[0] == len(
        real_traces
    ), (
        "Real embeddings should be of shape num_traces x embedding_shape! Got: %s"
        % str(real_embeddings.shape)
    )

    # check dissimilarity function
    for model_trace, model_embedding in zip(model_traces, model_embeddings):
        for real_trace, real_embedding in zip(model_traces, model_embeddings):
            dissimilarity = algorithm._calc_dissimilarity(
                model_embedding, real_embedding
            )
            assert 0 <= dissimilarity <= 1, "Dissimilarity values should be in [0,1]!"
            if model_trace == real_trace:
                assert (
                    dissimilarity == 0
                ), "Equal traces should have a dissimilarity of zero!"


def test_check_algorithm():
    algorithm, _, _, _ = example()
    check_algorithm(algorithm)


def test_algorithm_execution():
    algorithm, model_traces, real_traces, expected_matrix = example()
    result = algorithm().execute(model_traces, real_traces)
    matrix = result.get_dissimilarity_matrix()
    assert isinstance(
        matrix, np.ndarray
    ), "Dissimilarity matrix should be a numpy array!"
    assert matrix.dtype == np.float32, "Dissimilarity matrix should be of type float32!"
    assert np.all(matrix == expected_matrix), "Expected:\n%s\nGot:\n%s" % (
        str(expected_matrix),
        str(matrix),
    )
