from conformance_checking.__init__ import EmbeddingConformance
from conformance_checking.embedding.embedding_generator import Embedding_generator
import numpy as np
from typing import Dict, Tuple, List, Any


class EmbeddingConformance_ICT(EmbeddingConformance):
    """
    Inherit from EmbeddingConformance.
    Implement abstrackt methods _calc_embeddings and _calc_dissimilarity.
    Based on act2vec.
    Dissmilarity can be calculated using ICT.
    """

    @staticmethod
    def _calc_embeddings(
        model_traces: List[List[str]], real_traces: List[List[str]]
    ) -> Tuple[List[Any], List[Any], Any]:
        """Calculates the embeddings of the traces.

        :param model_traces: The traces coming from the model.
        :param real_traces: The traces coming from the real log.
        :return: Dicts for the model and real log contains
            index of activities and its frequencies
            and an implementation-specific context object.
        """

        emb_gen = Embedding_generator(
            model_traces + real_traces,
            trace2vec_windows_size=4,
            act2vec_windows_size=4,
            num_ns=4,
            activity_auto_train=False,
            trace_auto_train=False,
        )

        # start to train the models
        emb_gen.start_training()

        # return frequency tables for the model log and the real log
        # and an embedding lookup table
        model_embedding, real_embedding, context = emb_gen.get_activity_embedding(
            model_traces, real_traces
        )

        return (model_embedding, real_embedding, context)

    @staticmethod  # noqa: C901
    def _calc_dissimilarity(  # noqa: C901
        model_embedding: Dict[int, int],
        real_embedding: Dict[int, int],
        context: np.ndarray,
        k: int = 3,
    ) -> float:
        """calculates ICT between two embeddings.

        :param model_embedding: The first integer is the index of an activity,
            the second integer is the times that the activity shows in the trace
        :param real_embedding: The first integer is the index of an activity,
            the second integer is the times that the activity shows in the trace
        :param context: should be np.ndarray with dimension m x n,
            where n is the dimension of embedding, m is number of embeddings,
            context[i] is the embeddings of activity with index i
        :param k: number of edges considered per activity, defalt=3
        :return: the dissimiler of two traces as a floating-point value
        """

        vocab_len = len(context)

        # function: calculate normalized count of activity i within its trace
        def calc_d(embeddings: dict):
            d = np.zeros(vocab_len, dtype=np.double)
            # calculate the length of trace
            trace_len = 0
            for value in embeddings.values():
                trace_len += value

            for i in range(vocab_len):
                count = embeddings.get(i, 0)
                d[i] = count / trace_len
            return d

        d_model = calc_d(model_embedding)
        d_real = calc_d(real_embedding)

        # calculate Euclidean distance between embeddings word i and word j
        distance_matrix = np.zeros((vocab_len, vocab_len), dtype=np.double)
        for i in range(vocab_len):
            for j in range(vocab_len):
                if distance_matrix[i, j] != 0.0:
                    continue
                distance_matrix[i, j] = distance_matrix[j, i] = np.sqrt(
                    np.sum((context[i] - context[j]) ** 2)
                )

        # calculates ACT between two embeddings.
        dist = 0
        for i in range(0, len(d_model)):
            pi = d_model[i]  # the weight of the ith element in model trace
            # if this activity is not actually in model pi will be zero
            if pi == 0.0:
                continue
            dummy_s = np.argsort(
                distance_matrix[i]
            )  # have to change to only use the thing where q[j] != 0
            s = np.ones(k, dtype=int)
            it = 0
            j = 0
            while it < k and j < len(dummy_s):
                if d_real[dummy_s[j]] != 0.0:
                    s[it] = int(dummy_s[j])
                    it = it + 1
                j = j + 1
            l = 0  # noqa: E741
            while l < k and pi > 0:
                r = min(pi, d_real[s[l]])
                pi = pi - r
                dist = dist + r * distance_matrix[i, s[l]]
                l = l + 1  # noqa: E741
            if pi != 0:
                dist = dist + pi * distance_matrix[i, s[l - 1]]

        return dist
