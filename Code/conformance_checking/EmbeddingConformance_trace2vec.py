from conformance_checking.__init__ import EmbeddingConformance
from conformance_checking.embedding.embedding_generator import Embedding_generator
import numpy as np
from typing import Tuple, List, Any


class EmbeddingConformance_trace2vec(EmbeddingConformance):
    """
    Inherit from EmbeddingConformance.
    Implement abstrackt methods _calc_embeddings and _calc_dissimilarity.
    Based on trace2vec.
    Dissmilarity can be calculated using cosine distance.
    """

    @staticmethod
    def _calc_embeddings(
        model_traces: List[List[str]], real_traces: List[List[str]]
    ) -> Tuple[List[Any], List[Any], Any]:
        """Calculates the embeddings of the traces.

        :param model_traces: The traces coming from the model.
        :param real_traces: The traces coming from the real log.
        :return: two list contains trace embeddings for the model and real log.
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

        # return the trace embeddings of traces in the model log and real log
        return (emb_gen.get_trace_embedding(model_traces, real_traces), None)

    @staticmethod
    def _calc_dissimilarity(
        model_embedding: np.ndarray, real_embedding: np.ndarray, context=None
    ) -> float:

        return np.dot(model_embedding, real_embedding) / (
            np.linalg.norm(model_embedding) * np.linalg.norm(real_embedding)
        )
