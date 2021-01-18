from abc import ABC, abstractmethod
from typing import List, Tuple, Any

import numpy as np
import pm4py
import pickle


def import_xes(path_to_log_file):
    """Import an event log from a .xes file and return a List[List[str]],
    where the entry i,j is the j-th activity name of the i-th trace.
    :param path_to_log_file: a path to the log file to be imported
    :return: List[List[str]],where the entry i,j is the
    j-th activity name of the i-th trace.
    """
    event_log = pm4py.read_xes(path_to_log_file)

    return [[event["concept:name"] for event in trace] for trace in event_log]


def import_petri_net(path_to_model_file):
    """Import a petri net from a .pnml file.
    :param path_to_model_file: a path to the petri net file to be imported
    :return: a petri net, an initial marking and a final marking
    """
    net, initial_marking, final_marking = pm4py.read_petri_net(path_to_model_file)

    return net, initial_marking, final_marking


def generate_playout(net, initial_marking, final_marking):
    """Generate a playout given a petri net, initial_marking and final_marking
    and return a List[List[str]], where the entry i,j is the j-th activity
    name of the i-th trace.
    :param net: a petri net
    :param initial_marking: the initial marking of the petri net
    :param final_marking: the final marking of a petri net
    :return: a List[List[str]], where the entry i,j is the j-th activity name
    of the i-th trace.
    """

    playout_log = pm4py.simulation.playout.simulator.apply(
        net, initial_marking, final_marking=final_marking
    )

    return [[event["concept:name"] for event in trace] for trace in playout_log]


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

    def calc_fitness(self) -> float:
        fitness = np.average(self.get_dissimilarity_matrix().min(axis=0))
        return fitness

    def calc_precision(self) -> float:
        precision = np.average(self.get_dissimilarity_matrix().min(axis=1))
        return precision

    def save(self, path):
        pickle_file = open(path, "wb")
        pickle.dump(self.get_dissimilarity_matrix(), pickle_file)
        pickle_file.close()

    @staticmethod
    def load(path):
        pickle_file = open(path, "rb")
        matrix = pickle.load(pickle_file)
        pickle_file.close()
        return matrix



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
    ) -> Tuple[List[Any], List[Any]]:
        """Calculates the embeddings of the traces.

        :param model_traces: The traces coming from the model.
        :param real_traces: The traces coming from the real log.
        :return: The embeddings of the traces of the model and real log.
        """
        pass  # pragma: no cover

    @staticmethod
    @abstractmethod
    def _calc_dissimilarity(model_embedding: Any, real_embedding: Any) -> float:
        """Calculates the dissimilarity between two embeddings.

        :param model_embedding: the embedding of the model trace
        :param real_embedding: the embedding of the real trace
        :return: a floating-point value in [0, 1] where 1 is
        the maximum dissimilarity
        """
        pass  # pragma: no cover
