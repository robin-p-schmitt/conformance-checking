from conformance_checking import EmbeddingConformance


class Mock(EmbeddingConformance):
    """A mock for the base algorithm.

    The EmbeddingConformance class will be implemented by all the conformance
    checking algorithms we will implement. Each algorithm has to implement the
    two methods _calc_embeddings() and _calc_dissimilarity(). In this mock, we
    take the length of a trace as its embedding. The dissimilarity is then the
    relative difference in the length of a trace. This is a very basic implementation
    for demonstration purposes.
    """

    @staticmethod
    def _calc_embeddings(model_traces, real_traces):
        model_embeddings = [len(t) for t in model_traces]
        real_embeddings = [len(t) for t in real_traces]
        return model_embeddings, real_embeddings

    @staticmethod
    def _calc_dissimilarity(model_embedding, real_embedding):
        max_len = max(model_embedding, real_embedding)
        min_len = min(model_embedding, real_embedding)
        if max_len == 0:
            return 0
        else:
            return 1 - min_len / max_len


def main():
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
    print("Model traces: %s" % str(model_traces))
    print("Real traces: %s" % str(real_traces))
    print("Executing mocked algorithm...")
    dissimilarity_matrix = Mock().execute(model_traces, real_traces)

    print(
        "Dissimilarity matrix (entry at row i and column j is the distance of model "
        "trace i and real trace j):"
    )
    print(dissimilarity_matrix.get_dissimilarity_matrix())
    print(dissimilarity_matrix.calc_fitness())
    print(dissimilarity_matrix.calc_precision())


if __name__ == "__main__":
    main()
