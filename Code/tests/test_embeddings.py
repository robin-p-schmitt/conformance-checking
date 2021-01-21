from conformance_checking.embedding.embedding_generator import (
    Trace_Embedding_generator,
)

from sklearn.metrics.pairwise import cosine_similarity


def test_trace_embeddings():
    # example traces (need to be modified)
    trace1 = ["0"] * 5
    trace2 = ["1"] * 5
    trace3 = ["0"] * 3
    log = [trace1, trace2, trace3]

    # train trace2vec
    emb_gen = Trace_Embedding_generator(log, trace2vec_windows_size=4, auto_train=True)

    # get embeddings of the three traces
    embeddings, _ = emb_gen.get_trace_embedding(log, log)

    # compare every embedding with every other embedding
    sim = cosine_similarity(embeddings, embeddings)

    # similarity of trace 1 and trace 3
    sim1 = sim[2, 0]
    # similarity of trace 2 and trace 3
    sim2 = sim[2, 1]

    # want trace3 to be more similar to trace1 than to trace2
    assert sim1 > sim2
