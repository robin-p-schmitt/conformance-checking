from pyemd import emd
import numpy as np
from typing import Dict, Tuple


def calc_wmd(
    model_embedding: Dict[Tuple, int], real_embedding: Dict[Tuple, int]
) -> float:
    """calculates WMD between two embeddings.

    :param model_embedding: the embedding of the model trace
                            Dict[Tuple, int]: Tuple is the embedding of an activity, int is count of this activity within a trace
    :param real_embedding: the embedding of the real trace
                           Dict[Tuple, int]: Tuple is the embedding of an activity, int is count of this activity within a trace
    :return: a floating-point value
    """

    whole_embedding = set(list(model_embedding.keys()) + list(real_embedding.keys()))

    vocab_len = len(whole_embedding)

    # function: calculate normalized count of activity i within its trace
    def calc_d(embeddings: dict):
        d = np.zeros(vocab_len, dtype=np.double)
        # calculate the length of trace
        trace_len = 0
        for value in embeddings.values():
            trace_len += value

        for i, embedding in enumerate(whole_embedding):
            count = embeddings.get(embedding, 0)
            d[i] = count / trace_len
        return d

    d_model = calc_d(model_embedding)
    d_real = calc_d(real_embedding)

    # calculate Euclidean distance between embeddings word i and word j
    distance_matrix = np.zeros((vocab_len, vocab_len), dtype=np.double)
    for i, embedding1 in enumerate(whole_embedding):
        for j, embedding2 in enumerate(whole_embedding):
            if distance_matrix[i, j] != 0.0:
                continue
            distance_matrix[i, j] = distance_matrix[j, i] = np.sqrt(
                np.sum((np.array(embedding1) - np.array(embedding2)) ** 2)
            )

    dist = emd(d_model, d_real, distance_matrix)

    return dist


def ACT(p, q, C, k):
    """calculates ACT between two embeddings.

    :param p: a np.array contains normalized count of activity within a trace
    :param q: a np.array contains normalized count of activity within a trace
    :param C: dissimilar matrix of these two traces
    :param k: number of edges considered per activity
    :return: a floating-point value
    """

    t = 0
    for i in range(0, len(p)):
        pi = p[i]  # the weight of the ith element in p trace
        if pi == 0.0:  # if this activity is not actually in p pi will be zero
            continue
        dummy_s = np.argsort(
            C[i]
        )  # have to change to only use the thing where q[j] != 0
        s = np.ones(k, dtype=int)
        it = 0
        j = 0
        while it < k and j < len(dummy_s):
            if q[dummy_s[j]] != 0.0:
                s[it] = int(dummy_s[j])
                it = it + 1
            j = j + 1
        l = 0  # noqa: E741
        while l < k and pi > 0:
            r = min(pi, q[s[l]])
            pi = pi - r
            t = t + r * C[i, s[l]]
            l = l + 1  # noqa: E741
        if pi != 0:
            t = t + pi * C[i, s[l - 1]]
    return t


def calc_ict(
    model_embedding: Dict[Tuple, int], real_embedding: Dict[Tuple, int], k: int = 3
) -> float:
    """calculates ICT between two embeddings.

    :param model_embedding: the embedding of the model trace
                            Dict[Tuple, int]: Tuple is the embedding of an activity, int is count of this activity within a trace
    :param real_embedding: the embedding of the real trace
                           Dict[Tuple, int]: Tuple is the embedding of an activity, int is count of this activity within a trace
    :param k: number of edges considered per activity
    :return: a floating-point value
    """

    whole_embedding = set(list(model_embedding.keys()) + list(real_embedding.keys()))

    vocab_len = len(whole_embedding)

    # function: calculate normalized count of activity i within its trace
    def calc_d(embeddings: dict):
        d = np.zeros(vocab_len, dtype=np.double)
        # calculate the length of trace
        trace_len = 0
        for value in embeddings.values():
            trace_len += value

        for i, embedding in enumerate(whole_embedding):
            count = embeddings.get(embedding, 0)
            d[i] = count / trace_len
        return d

    d_model = calc_d(model_embedding)
    d_real = calc_d(real_embedding)

    # calculate Euclidean distance between embeddings word i and word j
    distance_matrix = np.zeros((vocab_len, vocab_len), dtype=np.double)
    for i, embedding1 in enumerate(whole_embedding):
        for j, embedding2 in enumerate(whole_embedding):
            if distance_matrix[i, j] != 0.0:
                continue
            distance_matrix[i, j] = distance_matrix[j, i] = np.sqrt(
                np.sum((np.array(embedding1) - np.array(embedding2)) ** 2)
            )

    dist = ACT(d_model, d_real, distance_matrix, k)

    return dist
