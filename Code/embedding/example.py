from embedding_generator import Embedding_generator
from pm4py.objects.log.importer.xes import importer as xes_importer

if __name__ == "__main__":
    # load a log with pm4py
    log = xes_importer.apply("logs/BPI_Challenge_2012.xes")
    # only keep the first 2000 traces, so it is faster.
    # If you want to test on the whole log, just remove the [:2000]
    log = [[event["concept:name"] for event in trace] for trace in log][:2000]

    # create instance of the embedding generator.
    # log: the log to train the embeddings on
    # trace2vec_window_size: number of context activities on each
    #                        side to predict target word
    # act2vec_window_size: number of positive samples for every activity
    # num_ns: number of negative samples for every positive sample in act2vec
    emb_gen = Embedding_generator(
        log, trace2vec_windows_size=4, act2vec_windows_size=4, num_ns=4
    )

    # for now, the embedding generator does not output the actual embeddings,
    # but just trains the models on the given log
    # the trace2vec produces an accuracy of 98% at the moment,
    # which we think has to be an error. We will investigate
    # this in the next sprint.
