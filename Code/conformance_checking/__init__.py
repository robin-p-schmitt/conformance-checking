from abc import ABC, abstractmethod
from typing import List, Tuple, Any
from conformance_checking.distances import calc_d

import numpy as np
import pm4py


def import_xes(path_to_log_file, key, limit=None):
    """Import an event log from a .xes file and return a List[List[str]],
    where the entry i,j is the j-th activity name of the i-th trace.
    :param path_to_log_file: a path to the log file to be imported
    :param key: activity name key for the given xes file
    :return: List[List[str]],where the entry i,j is the
    j-th activity name of the i-th trace.
    """
    event_log = pm4py.read_xes(path_to_log_file)

    return [[event[key] for event in trace] for trace in event_log][:limit]


def import_petri_net(path_to_model_file):
    """Import a petri net from a .pnml file.
    :param path_to_model_file: a path to the petri net file to be imported
    :return: a petri net, an initial marking and a final marking
    """
    net, initial_marking, final_marking = pm4py.read_petri_net(path_to_model_file)

    return net, initial_marking, final_marking


def generate_playout(net, initial_marking, final_marking, key):
    """Generate a playout given a petri net, initial_marking and final_marking
    and return a List[List[str]], where the entry i,j is the j-th activity
    name of the i-th trace.
    :param net: a petri net
    :param initial_marking: the initial marking of the petri net
    :param final_marking: the final marking of a petri net
    :param key: activity name key for the given petri net
    :return: a List[List[str]], where the entry i,j is the j-th activity name
    of the i-th trace.
    """

    playout_log = pm4py.simulation.playout.simulator.apply(
        net, initial_marking, final_marking=final_marking
    )

    return [[event[key] for event in trace] for trace in playout_log]


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
        fitness = 1 - np.average(self.get_dissimilarity_matrix().min(axis=0))
        return fitness

    def calc_precision(self) -> float:
        precision = 1 - np.average(self.get_dissimilarity_matrix().min(axis=1))
        return precision

    def save(self, path):
        np.save(path, self.get_dissimilarity_matrix())

    @staticmethod
    def load(path):
        matrix = np.load(path)
        return DissimilarityMatrix(matrix)


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

        def deduplicate(traces):
            known = {}
            deduplicated = []
            mapping = []
            for i, trace in enumerate(traces):
                trace_tuple = tuple(trace)
                if trace_tuple in known:
                    di = known[trace_tuple]
                else:
                    di = len(deduplicated)
                    deduplicated.append(trace)
                    known[trace_tuple] = di
                    mapping.append([])
                mapping[di].append(i)
            return deduplicated, mapping

        model_deduplicated, model_mappings = deduplicate(model_traces)
        real_deduplicated, real_mappings = deduplicate(real_traces)

        model_embeddings, real_embeddings, context = self._calc_embeddings(
            model_deduplicated, real_deduplicated
        )

        if context is not None:
            # algorithm of act2vec: need to calc d
            vocab_len = len(context)
            model_embeddings = calc_d(model_embeddings, vocab_len)
            real_embeddings = calc_d(real_embeddings, vocab_len)

        dissimilarity_matrix = np.zeros(
            (len(model_traces), len(real_traces)), dtype=np.float32
        )
        for model_embedding, model_map in zip(model_embeddings, model_mappings):
            for real_embedding, real_map in zip(real_embeddings, real_mappings):
                dissimilarity = self._calc_dissimilarity(
                    model_embedding, real_embedding, context
                )
                for i in model_map:
                    for j in real_map:
                        dissimilarity_matrix[i, j] = dissimilarity
        return DissimilarityMatrix(dissimilarity_matrix)

    @abstractmethod
    def _calc_embeddings(
        self, model_traces: List[List[str]], real_traces: List[List[str]]
    ) -> Tuple[List[Any], List[Any], Any]:
        """Calculates the embeddings of the traces.

        :param model_traces: The traces coming from the model.
        :param real_traces: The traces coming from the real log.
        :return: The embeddings of the traces of the model and real log
        and an implementation-specific context object.
        """
        pass  # pragma: no cover

    @abstractmethod
    def _calc_dissimilarity(
        self, model_embedding: Any, real_embedding: Any, context: Any
    ) -> float:
        """Calculates the dissimilarity between two embeddings.

        :param model_embedding: the embedding of the model trace
        :param real_embedding: the embedding of the real trace
        :param context: the context object (implementation specific)
        :return: a floating-point value in [0, 1] where 1 is
        the maximum dissimilarity
        """
        pass  # pragma: no cover
