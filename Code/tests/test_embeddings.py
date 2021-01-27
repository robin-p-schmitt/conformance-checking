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


def test_embeddings_generator():
    trace2vec_gen_trained = TraceEmbeddingGenerator(log, auto_train=True)
    act2vec_gen_trained = ActivityEmbeddingGenerator(log, auto_train=True)

    # assert accuracy of trace2vec
    trace2vec_acc = trace2vec_gen_trained.evaluate_model()

    # assert accuracy of act2vec
    act2vec_acc = act2vec_gen_trained.evaluate_model()

    # assert type and shape of trace embeddings
    test_log = log[:5]
    trace_embeddings, _ = trace2vec_gen_trained.get_trace_embedding(test_log, test_log)
    _, _, act_embeddings = act2vec_gen_trained.get_activity_embedding(
        test_log, test_log
    )

    assert (
        trace2vec_acc > 0.85
        and act2vec_acc > 0.75
        and (
            type(trace_embeddings) == np.ndarray and trace_embeddings.shape == (5, 128)
        )
        and (
            type(act_embeddings) == np.ndarray
            and act_embeddings.shape
            == (
                len(activities) + 1,
                128,
            )
        )
    )


def test_exceptions():
    assert str(ModelNotTrainedError("test")) == "ModelNotTrainedError, test"
    assert str(ModelNotTrainedError()) == "ModelNotTrainedError has been raised"

    trace2vec_gen_not_trained = TraceEmbeddingGenerator(log)
    act2vec_gen_not_trained = ActivityEmbeddingGenerator(log)

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
        trace2vec_gen_not_trained.evaluate_model()

    with pytest.raises(
        ModelNotTrainedError, match="model for trace embeddings is not trained yet"
    ):
        trace2vec_gen_not_trained.get_trace_embedding([], [])
