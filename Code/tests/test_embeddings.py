from conformance_checking.embedding.embedding_generator import (
    ActivityEmbeddingGenerator,
    TraceEmbeddingGenerator,
    ModelNotTrainedError,
)

import numpy as np
from conformance_checking import import_xes
import os
import pytest

absPath = os.path.abspath(__file__)
fileDir = os.path.dirname(absPath)
code = os.path.dirname(fileDir)
data = os.path.join(code, "data")
log = import_xes(os.path.join(data, "BPI_Challenge_2012.xes"))
log = log[:1000]
activities = set([act for trace in log for act in trace])

trace2vec_gen_trained = TraceEmbeddingGenerator(log, auto_train=True)
act2vec_gen_trained = ActivityEmbeddingGenerator(log, auto_train=True)

trace2vec_gen_not_trained = TraceEmbeddingGenerator(log)
act2vec_gen_not_trained = ActivityEmbeddingGenerator(log)


# assert accuracy of trace2vec
def test_trace2vec_acc():
    acc = trace2vec_gen_trained.evaluate_model()
    assert acc > 0.85


# assert accuracy of act2vec
def test_act2vec_acc():
    acc = act2vec_gen_trained.evaluate_model()
    assert acc > 0.75


# assert type and shape of trace embeddings
def test_trace2vec_emb():
    test_log = log[:5]
    embeddings, _ = trace2vec_gen_trained.get_trace_embedding(test_log, test_log)

    assert type(embeddings) == np.ndarray and embeddings.shape == (5, 128)


# assert type and shape of activity embeddings
def test_act2vec_emb():
    test_log = log[:5]
    _, _, embeddings = act2vec_gen_trained.get_activity_embedding(test_log, test_log)

    assert type(embeddings) == np.ndarray and embeddings.shape == (
        len(activities) + 1,
        128,
    )


def test_exceptions():
    with pytest.raises(
        ModelNotTrainedError, match="model for activity embeddings is not trained yet"
    ):
        act2vec_gen_not_trained.evaluate_model()

    with pytest.raises(
        ModelNotTrainedError, match="model for activity embeddings is not trained yet"
    ):
        act2vec_gen_not_trained.get_activity_embedding([], [])

    with pytest.raises(
        ModelNotTrainedError, match="model for trace embeddings is not trained yet"
    ):
        trace2vec_gen_not_trained.get_trace_embedding([], [])

    with pytest.raises(
        ModelNotTrainedError, match="model for trace embeddings is not trained yet"
    ):
        trace2vec_gen_not_trained.get_trace_embedding([], [])
