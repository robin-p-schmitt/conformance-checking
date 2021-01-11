from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class DissimilarityMatrix:
    """The result returned by EmbeddingConformance.execute().

    This class contains the dissimilarity matrix and provides methods for I/O and
    fitness and precision calculation.
    """

    def __init__(self, dissimilarity_matrix: np.ndarray):
        """Create a new dissimilarity matrix from a NumPy array

        :param dissimilarity_matrix: the NumPy dissimilarity matrix
        """
        self._dissimilarity_matrix = dissimilarity_matrix

    def get_dissimilarity_matrix(self) -> np.ndarray:
        """
        :return: the dissimilarity matrix where an entry at [i,j] is the dissimilarity
        between model trace i and real trace j.
        """
        return self._dissimilarity_matrix

    def calc_fitness(self):
        raise NotImplementedError

    def calc_precision(self):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    @staticmethod
    def load(path):
        raise NotImplementedError


class EmbeddingConformance(ABC):
    """Abstract base class for conformance checking algorithms based on embeddings."""

    def execute(
        self, model_traces: List[List[str]], real_traces: List[List[str]]
    ) -> DissimilarityMatrix:
        """Executes this algorithm on the given traces.

        This method is stateless, i.e. no changes are made to `self`.
        It is not static to allow for overriding the abstract methods.

        :param model_traces: The traces coming from the model.
        Rows of the dissimilarity matrix.
        :param real_traces: The traces coming from the real log.
        Columns of the dissimilarity matrix.
        :return: The dissimilarity matrix.
        """

        model_embeddings, real_embeddings = self._calc_embeddings(
            model_traces, real_traces
        )
        dissimilarity_matrix = np.zeros(
            (len(model_traces), len(real_traces)), dtype=np.float32
        )
        for i, model_embedding in enumerate(model_embeddings):
            for j, real_embedding in enumerate(real_embeddings):
                dissimilarity_matrix[i, j] = self._calc_dissimilarity(
                    model_embedding, real_embedding
                )
        return DissimilarityMatrix(dissimilarity_matrix)

    @staticmethod
    @abstractmethod
    def _calc_embeddings(
        model_traces: List[List[str]], real_traces: List[List[str]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates the embeddings of the traces.

        :param model_traces: The traces coming from the model.
        :param real_traces: The traces coming from the real log.
        :return: The NumPy embedding matrices for the model and real log.
        Each is of shape num_traces x embedding_shape.
        """
        pass  # pragma: no cover

    @staticmethod
    @abstractmethod
    def _calc_dissimilarity(
        model_embedding: np.ndarray, real_embedding: np.ndarray
    ) -> float:
        """Calculates the dissimilarity between two embeddings.

        :param model_embedding: the embedding of the model trace
        :param real_embedding: the embedding of the real trace
        :return: a floating-point value in [0, 1] where 1 is
        the maximum dissimilarity
        """
        pass  # pragma: no cover
