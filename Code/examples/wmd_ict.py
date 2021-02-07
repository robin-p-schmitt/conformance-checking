from conformance_checking.distances import calc_wmd, calc_ict, calc_euclidean, _calc_d
import numpy as np


def main():
    # create some embeddings as example
    # (int, int, ...): int =
    # embedding of a activity: count of this activity within a trace
    model_embedding = [{0: 3, 1: 1, 2: 2}]
    real_embedding = [{0: 2}]
    context = np.array([[0.4, 0.3], [0.2, 0.6], [0.5, 0.9]])

    # calculate Euclidean distance matrix
    distance_matrix = calc_euclidean(context)

    # calc d for embeddings
    vocab_len = len(context)
    d_model = _calc_d(model_embedding, vocab_len)
    d_real = _calc_d(real_embedding, vocab_len)

    # calculate WMD between these two traces
    print("WMD: ", calc_wmd(d_model[0], d_real[0], distance_matrix))
    print("ICT: ", calc_ict(d_model[0], d_real[0], distance_matrix))


if __name__ == "__main__":
    main()
