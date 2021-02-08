from conformance_checking.embedding.embedding_generator import (
    ActivityEmbeddingGenerator,
    TraceEmbeddingGenerator,
)

from conformance_checking import import_xes
import os

if __name__ == "__main__":
    # load a log with our importer
    absPath = os.path.abspath(__file__)
    fileDir = os.path.dirname(absPath)
    code = os.path.dirname(fileDir)
    data = os.path.join(code, "data")
    log = import_xes(os.path.join(data, "BPI_Challenge_2012.xes"), "concept:name", limit=2000)

    """
    example for generating each activity and trace embeddings
    """

    act_emb_gen = ActivityEmbeddingGenerator(log, act2vec_windows_size=4, num_ns=4)
    # start training manually
    act_emb_gen.start_training()

    trace_emb_gen = TraceEmbeddingGenerator(
        log, trace2vec_windows_size=4, auto_train=True
    )

    # create example model and real log
    model_log = log[:3]
    real_log = log[3:8]

    # get frequency tables for the model log and the real log
    # and an embedding lookup table
    model_freq, real_freq, embeddings = act_emb_gen.get_activity_embedding(
        model_log, real_log
    )

    print(
        "\nThe frequency of activity with index 10 in the",
        "first trace from model_log: {}\n".format(model_freq[0][10]),
    )
    print(
        "A list of dictionaries containing the counts of",
        "activities in traces from the real log: \n{}\n".format(real_freq),
    )
    print("The embedding of the activity with index 0: \n{}\n".format(embeddings[0]))

    # get the trace embeddings of traces in the model log and real log
    model_emb, real_emb = trace_emb_gen.get_trace_embedding(model_log, real_log)

    print(
        "The embedding of the first trace in the model log: \n{}".format(model_emb[0])
    )
