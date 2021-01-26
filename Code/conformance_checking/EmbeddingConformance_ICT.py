from conformance_checking.__init__ import EmbeddingConformance
from conformance_checking.distances import calc_ict
from conformance_checking.embedding.embedding_generator import Embedding_generator
import numpy as np
from typing import Dict, Tuple, List, Any


class EmbeddingConformance_ICT(EmbeddingConformance):
    """
    Inherit from EmbeddingConformance.
    Implement abstrackt methods _calc_embeddings and _calc_dissimilarity.
    Based on act2vec.
    Dissimilarity can be calculated using ICT.
    :param k: number of edges considered per activity, default=3
    """

    def __init__(self, k=3):
        self.k = k
        print(k)

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
        emb_gen.activity_embedding_generator.start_training()

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
        :param k: number of edges considered per activity, default=3
        :return: the dissimilarity of two traces as a floating-point value
        """

        return calc_ict(model_embedding, real_embedding, context, k)
