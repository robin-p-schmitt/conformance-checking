from conformance_checking.embedding.embedding_generator import (
    EmbeddingGenerator,
    ActivityEmbeddingGenerator,
    TraceEmbeddingGenerator,
)
from pm4py.objects.log.importer.xes import importer as xes_importer

if __name__ == "__main__":
    # load a log with pm4py
    variant = xes_importer.Variants.ITERPARSE
    parameters = {variant.value.Parameters.MAX_TRACES: 2000}
    log = xes_importer.apply(
        "data/BPI_Challenge_2012.xes", variant=variant, parameters=parameters
    )

    # get log as format List[List[str]]
    log = [[event["concept:name"] for event in trace] for trace in log]

    # example code for generating activity and trace embeddings at the same time
    # this will raise error since start_training() function was not called

    # create instance of the embedding generator.
    # log: the log to train the embeddings on
    # trace2vec_window_size: number of context activities on each
    #                        side to predict target word
    # act2vec_window_size: number of positive samples for every activity
    # num_ns: number of negative samples for every positive sample in act2vec
    emb_gen = EmbeddingGenerator(
        log,
        trace2vec_windows_size=4,
        act2vec_windows_size=4,
        num_ns=4,
        activity_auto_train=False,
        trace_auto_train=False,
    )

    # start to train the models
    emb_gen.start_training()

    # create example model and real log
    model_log = log[:3]
    real_log = log[3:8]

    # get frequency tables for the model log and the real log
    # and an embedding lookup table
    model_freq, real_freq, embeddings = emb_gen.get_activity_embedding(
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
    model_emb, real_emb = emb_gen.get_trace_embedding(model_log, real_log)

    print(
        "The embedding of the first trace in the model log: \n{}".format(model_emb[0])
    )

    """
    example for generating each activity and trace embeddings seperately
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
