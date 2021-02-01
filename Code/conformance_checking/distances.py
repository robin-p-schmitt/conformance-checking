from pyemd import emd
import numpy as np
from typing import Dict, List, Any


def _handle_empty_traces(d_model, d_real):
    if (True in np.isnan(d_model)) and (True in np.isnan(d_real)):
        return 0.0
    else:
        return 1.0


def calc_euclidean(context: np.ndarray) -> np.ndarray:
    """calculates Euclidean distances between activities in logs.

    :param context: should be np.ndarray with dimension m x n,
        where n is the dimension of embedding, m is number of embeddings,
        context[i] is the embeddings of activity with index i
    :return: the dissimilarity matrix for every two activities in logs.
    """

    act_num = len(context)
    distance_matrix = np.zeros((act_num, act_num), dtype=np.double)

    for i in range(act_num):
        for j in range(act_num):
            if distance_matrix[i, j] != 0.0:
                continue
            distance_matrix[i, j] = distance_matrix[j, i] = np.sqrt(
                np.sum((context[i] - context[j]) ** 2)
            )

    return distance_matrix


def calc_d(embeddings: List[Dict[int, int]], vocab_len: int) -> np.ndarray:
    """calculates d for a trace

    :param embeddings: Keys of dict is the index of an activity,
        values of dict is the times that the activity shows in the trace
    :param vocab_len: number of activities
    :return: d of the input trace
    """

    d = np.zeros((len(embeddings), vocab_len), dtype=np.double)

    for i, embedding in enumerate(embeddings):
        # calculate the length of trace
        trace_len = 0
        for value in embedding.values():
            trace_len += value

        if trace_len == 0:
            # for empty trace, we set d as np.nan
            d[i] = np.nan
        else:
            for j in range(vocab_len):
                count = embedding.get(j, 0)
                d[i, j] = count / trace_len

    return d


def calc_wmd(
    d_model: np.ndarray, d_real: np.ndarray, distance_matrix: np.ndarray
) -> float:
    """calculates WMD between two embeddings.

    :param d_model: d of a model trace
    :param d_real: d of a real trace
    :param distance_matrix: a distance matrix for Euclidean distances of
        every two actives in all traces
    :return: the dissimilarity of two traces as a floating-point value
    """

    if (True in np.isnan(d_model)) or (True in np.isnan(d_real)):
        return _handle_empty_traces(d_model, d_real)

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
    d_model: np.ndarray,
    d_real: np.ndarray,
    distance_matrix: np.ndarray,
    k: int = 3,
) -> float:
    """calculates ICT between two embeddings.

    :param d_model: d of a model trace
    :param d_real: d of a real trace
    :param distance_matrix: a distance matrix for Euclidean distances of
        every two actives in all traces
    :return: the dissimilarity of two traces as a floating-point value
    """

    if (True in np.isnan(d_model)) or (True in np.isnan(d_real)):
        return _handle_empty_traces(d_model, d_real)

    dist = ACT(d_model, d_real, distance_matrix, k)

    return dist
