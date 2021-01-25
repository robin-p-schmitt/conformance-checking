from conformance_checking.EmbeddingConformance_ICT import (
    EmbeddingConformance_ICT,
)


def main():
    model_traces = [
        ["hi", "foo"],
        ["hi", "foo"],
        ["bar"],
        ["a", "long", "trace", "with", "doubled", "words", "like", "long"],
    ]
    real_traces = [
        ["foobar", "hi"],
        ["bar"],
        ["bar"],
        ["a", "long", "long", "trace", "but", "not", "the", "same"],
    ]
    print("Model traces: %s" % str(model_traces))
    print("Real traces: %s" % str(real_traces))

    print("Executing EmbeddingConformance_act2vec algorithm...")
    dissimilarity_matrix = EmbeddingConformance_ICT().execute(model_traces, real_traces)

    print("Dissimilar matrix: ")
    print(print(dissimilarity_matrix.get_dissimilarity_matrix()))


if __name__ == "__main__":
    main()
