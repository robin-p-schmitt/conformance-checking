from conformance_checking.embedding.generate_training_data import (
    generate_activity_vocab,
    generate_trace_vocab,
    generate_act2vec_training_data,
    generate_trace2vec_training_data,
)
import os
import numpy as np
from conformance_checking import import_xes

absPath = os.path.abspath(__file__)
fileDir = os.path.dirname(absPath)
code = os.path.dirname(fileDir)
data = os.path.join(code, "data")
log = import_xes(os.path.join(data, "BPI_Challenge_2012.xes"), "concept:name")
log = log[:50]

unique_activities = set([act for trace in log for act in trace])
unique_traces = np.unique(log)


def test_act2vec_training_data():
    # assert if activity vocab contains all unique activities + padding token
    act_vocab = generate_activity_vocab(log)
    assert len(act_vocab) == len(unique_activities) + 1

    window_size = 3
    num_ns = 4
    targets, contexts, labels = generate_act2vec_training_data(
        log, act_vocab, window_size, num_ns
    )

    # assert that targets, contexts and labels are of same length
    assert len(targets) == len(contexts)
    assert len(contexts) == len(labels)
    # assert that contexts contain num_ns negative samples + 1 positive sample
    assert len(contexts[0]) == num_ns + 1


def test_trace2vec_training_data():
    # assert if activity vocab contains all unique activities + padding token
    act_vocab = generate_activity_vocab(log)
    assert len(act_vocab) == len(unique_activities) + 1

    # assert if trace vocab contains all unique traces
    trace_vocab = generate_trace_vocab(log, act_vocab)
    assert len(trace_vocab) == len(unique_traces)

    window_size = 3
    traces, contexts, labels = generate_trace2vec_training_data(
        log, act_vocab, trace_vocab, window_size
    )

    # assert that traces, contexts and labels are of same length
    assert len(traces) == len(contexts)
    assert len(contexts) == len(labels)
    assert len(contexts[0]) == window_size * 2
    # label should be activity as one-hot-vector
    assert len(labels[0]) == len(act_vocab)
