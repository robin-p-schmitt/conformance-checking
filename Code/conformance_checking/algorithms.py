from conformance_checking.__init__ import EmbeddingConformance
from conformance_checking.distances import calc_wmd, calc_ict
from conformance_checking.embedding.embedding_generator import (
    ActivityEmbeddingGenerator,
    TraceEmbeddingGenerator,
)
import numpy as np
from typing import Dict, Tuple, List, Any


class Act2VecWmdConformance(EmbeddingConformance):
    """
    Inherit from EmbeddingConformance.
    Implement abstract methods _calc_embeddings and _calc_dissimilarity.
    Based on act2vec.
    Dissimilarity can be calculated using WMD.
    """

    def __init__(
        self,
        window_size=2,
        num_negative=2,
        num_epochs=10,
        batch_size=1024,
        embedding_size=128,
    ):
        super(Act2VecWmdConformance, self).__init__()
        self.window_size = window_size
        self.num_negative = num_negative
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.embedding_size = embedding_size

    def _calc_embeddings(
        self, model_traces: List[List[str]], real_traces: List[List[str]]
    ) -> Tuple[List[Any], List[Any], Any]:
        """Calculates the embeddings of the traces.

        :param model_traces: The traces coming from the model.
        :param real_traces: The traces coming from the real log.
        :return: Dicts for the model and real log contains
            index of activities and its frequencies
            and an implementation-specific context object.
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
        return emb_gen.get_activity_embedding(model_traces, real_traces)

    def _calc_dissimilarity(
        self,
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
        :return: the dissimilarity of two traces as a floating-point value
        """

        return calc_wmd(model_embedding, real_embedding, context)


class Act2VecIctConformance(EmbeddingConformance):
    """
    Inherit from EmbeddingConformance.
    Implement abstract methods _calc_embeddings and _calc_dissimilarity.
    Based on act2vec.
    Dissimilarity can be calculated using ICT.
    :param k: number of edges considered per activity, default=3
    """

    def __init__(
        self,
        k=3,
        window_size=2,
        num_negative=2,
        num_epochs=10,
        batch_size=1024,
        embedding_size=128,
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
    ) -> Tuple[List[Any], List[Any], Any]:
        """Calculates the embeddings of the traces.

        :param model_traces: The traces coming from the model.
        :param real_traces: The traces coming from the real log.
        :return: Dicts for the model and real log contains
            index of activities and its frequencies
            and an implementation-specific context object.
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
            model_traces, real_traces
        )

        return model_embedding, real_embedding, context

    def _calc_dissimilarity(
        self,
        model_embedding: Dict[int, int],
        real_embedding: Dict[int, int],
        context: np.ndarray,
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

        return calc_ict(model_embedding, real_embedding, context, k=self.k)


class Trace2VecCosineConformance(EmbeddingConformance):
    """
    Inherit from EmbeddingConformance.
    Implement abstrackt methods _calc_embeddings and _calc_dissimilarity.
    Based on trace2vec.
    Dissimilarity can be calculated using cosine distance.
    """

    def __init__(
        self, window_size=2, num_epochs=10, batch_size=1024, embedding_size=128
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
