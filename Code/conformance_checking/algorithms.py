from conformance_checking.__init__ import EmbeddingConformance
from conformance_checking.distances import calc_wmd, calc_ict, calc_euclidean, _calc_d
from conformance_checking.embedding.embedding_generator import (
    ActivityEmbeddingGenerator,
    TraceEmbeddingGenerator,
)
import numpy as np
from typing import Tuple, List, Any


class Act2VecWmdConformance(EmbeddingConformance):
    """Implements :py:class:`conformance_checking.EmbeddingConformance` using act2vec
    and WMD.

    First, the activities are extracted from the traces and an act2vec embedding is
    calculated. Second, every trace gets replaced by the normalized count of activities.
    Third, the final distance is calculated using word mover distance by using the
    euclidean distances between activities.

    The embedding format are the d values from the original paper and the context object
    is a distance matrix between all pairs of activities.
    """

    def __init__(
        self,
        window_size=3,
        num_negative=5,
        num_epochs=100,
        batch_size=1024,
        embedding_size=16,
    ):
        super(Act2VecWmdConformance, self).__init__()
        self.window_size = window_size
        self.num_negative = num_negative
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.embedding_size = embedding_size

    def _calc_embeddings(
        self, model_traces: List[List[str]], real_traces: List[List[str]]
    ) -> Tuple[np.ndarray, np.ndarray, Any]:
        """Calculates the embeddings of the traces.

        :param model_traces: The traces coming from the model.
        :param real_traces: The traces coming from the real log.
        :return: Dicts for the model and real log contains
            index of activities and its frequencies
            and a distance matrix for Euclidean distances of
            every two actives in all traces.
        """

        emb_gen = ActivityEmbeddingGenerator(
            model_traces + real_traces,
            act2vec_windows_size=self.window_size,
            num_ns=self.num_negative,
            auto_train=False,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            embedding_size=self.embedding_size,
        )

        # start to train the models
        emb_gen.start_training()

        model_embedding, real_embedding, context = emb_gen.get_activity_embedding(
            model_traces, real_traces, norm=True
        )
        dist_matrix = calc_euclidean(context)
        model_embedding = _calc_d(model_embedding, len(dist_matrix))
        real_embedding = _calc_d(real_embedding, len(dist_matrix))

        # return frequency tables for the model log and the real log
        # and an embedding lookup table
        return model_embedding, real_embedding, dist_matrix

    def _calc_dissimilarity(
        self, d_model: np.ndarray, d_real: np.ndarray, distance_matrix: np.ndarray
    ) -> float:
        """calculates WMD between two embeddings.

        :param d_model: d of a model trace
        :param d_real: d of a real trace
        :param distance_matrix: a distance matrix for Euclidean distances of
            every two actives in all traces
        :return: the dissimilarity of two traces as a floating-point value
        """

        return calc_wmd(d_model, d_real, distance_matrix)


class Act2VecIctConformance(EmbeddingConformance):
    """Implements :py:class:`conformance_checking.EmbeddingConformance` using act2vec
    and ICT.

    First, the activities are extracted from the traces and an act2vec embedding is
    calculated. Second, every trace gets replaced by the normalized count of activities.
    Third, the final distance is calculated using iterative constrained transfers by
    using the euclidean distances between activities.

    .. note:: This method should be faster than :py:class:`Act2VecWmdConformance`
        while being slightly less accurate.

    The embedding format are the d values from the original paper and the context object
    is a distance matrix between all pairs of activities.
    """

    def __init__(
        self,
        k=3,
        window_size=3,
        num_negative=5,
        num_epochs=100,
        batch_size=1024,
        embedding_size=16,
    ):
        super(Act2VecIctConformance, self).__init__()
        self.k = k
        self.window_size = window_size
        self.num_negative = num_negative
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.embedding_size = embedding_size

    def _calc_embeddings(
        self, model_traces: List[List[str]], real_traces: List[List[str]]
    ) -> Tuple[np.ndarray, np.ndarray, Any]:
        """Calculates the embeddings of the traces.

        :param model_traces: The traces coming from the model.
        :param real_traces: The traces coming from the real log.
        :return: Dicts for the model and real log contains
            index of activities and its frequencies
            and a distance matrix for Euclidean distances of
            every two actives in all traces.
        """

        emb_gen = ActivityEmbeddingGenerator(
            model_traces + real_traces,
            act2vec_windows_size=self.window_size,
            num_ns=self.num_negative,
            auto_train=False,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            embedding_size=self.embedding_size,
        )

        # start to train the models
        emb_gen.start_training()

        # return frequency tables for the model log and the real log
        # and an embedding lookup table
        model_embedding, real_embedding, context = emb_gen.get_activity_embedding(
            model_traces, real_traces, norm=True
        )
        dist_matrix = calc_euclidean(context)
        model_embedding = _calc_d(model_embedding, len(dist_matrix))
        real_embedding = _calc_d(real_embedding, len(dist_matrix))

        return model_embedding, real_embedding, dist_matrix

    def _calc_dissimilarity(
        self, d_model: np.ndarray, d_real: np.ndarray, distance_matrix: np.ndarray
    ) -> float:
        """calculates ICT between two embeddings.

        :param d_model: d of a model trace
        :param d_real: d of a real trace
        :param distance_matrix: a distance matrix for Euclidean distances of
            every two actives in all traces
        :return: the dissimilarity of two traces as a floating-point value
        """

        return calc_ict(d_model, d_real, distance_matrix, k=self.k)


class Trace2VecCosineConformance(EmbeddingConformance):
    """Implements :py:class:`conformance_checking.EmbeddingConformance` using trace2vec
    and cosine distances.

    First, the traces are embedded using trace2vec. Second, the distance between every
    two traces is calculated by taking the cosine distance between their embeddings.

    .. note:: This method should is fastest, but produces different results than the
        act2vec methods.

    The embedding format are embedding vectors. There is no context object.
    """

    def __init__(
        self, window_size=3, num_epochs=300, batch_size=1024, embedding_size=16
    ):
        super(Trace2VecCosineConformance, self).__init__()
        self.window_size = window_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.embedding_size = embedding_size

    def _calc_embeddings(
        self, model_traces: List[List[str]], real_traces: List[List[str]]
    ) -> Tuple[List[Any], List[Any], Any]:
        """Calculates the embeddings of the traces.

        :param model_traces: The traces coming from the model.
        :param real_traces: The traces coming from the real log.
        :return: two list contains trace embeddings for the model and real log.
        """

        emb_gen = TraceEmbeddingGenerator(
            model_traces + real_traces,
            trace2vec_windows_size=self.window_size,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            embedding_size=self.embedding_size,
        )

        # start to train the models
        emb_gen.start_training()

        model_embeddings, real_embeddings = emb_gen.get_trace_embedding(
            model_traces, real_traces
        )

        # return the trace embeddings of traces in the model log and real log
        return model_embeddings, real_embeddings, None

    def _calc_dissimilarity(
        self, model_embedding: np.ndarray, real_embedding: np.ndarray, context=None
    ) -> float:

        cosine = np.dot(model_embedding, real_embedding) / (
            np.linalg.norm(model_embedding) * np.linalg.norm(real_embedding)
        )
        # scale the values to [0, 1]
        # the max is necessary to prevent -0.0 values
        return max(0, 1 - (cosine + 1) / 2)
