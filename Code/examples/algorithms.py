from conformance_checking.algorithms import (
    Act2VecWmdConformance,
    Act2VecIctConformance,
    Trace2VecCosineConformance,
)

import numpy as np


def main():
    model_traces = [
        ["hi", "foo"],
        ["hi", "foo"],
        ["bar"],
        [],
        ["a", "long", "trace", "with", "doubled", "words", "like", "long"],
    ]
    real_traces = [
        ["foobar", "hi"],
        ["bar"],
        ["bar"],
        [],
        ["a", "long", "long", "trace", "but", "not", "the", "same"],
    ]
    print("Model traces: %s" % str(model_traces))
    print("Real traces: %s" % str(real_traces))

    print("Executing Act2VecWmdConformance algorithm...")
    dissimilarity_matrix_a = Act2VecWmdConformance().execute(model_traces, real_traces)

    print("Executing Act2VecIctConformance algorithm...")
    dissimilarity_matrix_b = Act2VecIctConformance().execute(model_traces, real_traces)

    print("Executing Trace2VecCosineConformance algorithm...")
    dissimilarity_matrix_c = Trace2VecCosineConformance().execute(
        model_traces, real_traces
    )

    np.set_printoptions(precision=4, suppress=True)  # similar effect to %.4f

    print_results("Act2VecWmdConformance", dissimilarity_matrix_a)
    print_results("Act2VecIctConformance", dissimilarity_matrix_b)
    print_results("Trace2VecCosineConformance", dissimilarity_matrix_c)


def print_results(name, dissimilarity_matrix):
    print("Dissimilarity matrix of %s: " % name)
    print(dissimilarity_matrix.get_dissimilarity_matrix())
    print("Precision: %.4f" % dissimilarity_matrix.calc_precision())
    print("Fitness: %.4f" % dissimilarity_matrix.calc_fitness())


if __name__ == "__main__":
    main()
