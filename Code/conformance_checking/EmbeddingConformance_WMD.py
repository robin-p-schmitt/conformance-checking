from conformance_checking.__init__ import EmbeddingConformance
from conformance_checking.embedding.embedding_generator import Embedding_generator
from pyemd import emd
import numpy as np
from typing import Dict, Tuple, List, Any


class EmbeddingConformance_WMD(EmbeddingConformance):
    """
    Inherit from EmbeddingConformance.
    Implement abstrackt methods _calc_embeddings and _calc_dissimilarity.
    Based on act2vec.
    Dissmilarity can be calculated using WMD.
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
        return emb_gen.get_activity_embedding(model_traces, real_traces)

    @staticmethod
    def _calc_dissimilarity(
        model_embedding: Dict[int, int],
        real_embedding: Dict[int, int],
        context: np.ndarray,
    ) -> float:
        """calculates WMD between two embeddings.

        :param model_embedding: The first integer is the index of an activity,
            the second integer is the times that the activity shows in the trace
        :param real_embedding: The first integer is the index of an activity,
            the second integer is the times that the activity shows in the trace
        :param context: should be np.ndarray with dimension m x n,
            where n is the dimension of embedding, m is number of embeddings,
            context[i] is the embeddings of activity with index i
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

        dist = emd(d_model, d_real, distance_matrix)

        return dist
