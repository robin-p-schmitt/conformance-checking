from conformance_checking.embedding.embedding_generator import (
    Activity_Embedding_generator,
    Trace_Embedding_generator,
)

import numpy as np
from pm4py.objects.log.importer.xes import importer as xes_importer
import os

absPath = os.path.abspath(__file__)
fileDir = os.path.dirname(absPath)
code = os.path.dirname(fileDir)
data = os.path.join(code, "data")

variant = xes_importer.Variants.ITERPARSE
parameters = {variant.value.Parameters.MAX_TRACES: 1000}
log = xes_importer.apply(
    os.path.join(data, "BPI_Challenge_2012.xes"), variant=variant, parameters=parameters
)
log = [[event["concept:name"] for event in trace] for trace in log]
activities = set([act for trace in log for act in trace])

trace2vec_gen_trained = Trace_Embedding_generator(log, auto_train=True)
act2vec_gen_trained = Activity_Embedding_generator(log, auto_train=True)

trace2vec_gen_not_trained = Trace_Embedding_generator(log)
act2vec_gen_not_trained = Activity_Embedding_generator(log)


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
