from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Union

import numpy as np
import pm4py


def import_xes(path_to_log_file: str, key: str, limit=None) -> List[List[str]]:
    """Import an event log from a .xes file and return a List[List[str]],
    where the entry i,j is the j-th activity name of the i-th trace.

    :param path_to_log_file: A path to the log file to be imported
    :type path_to_log_file: str
    :param key: Activity name key for the given xes file
    :type key: str
    :param limit: The maximum number of traces to load.
    :type limit: int
    :return: List[List[str]], where the entry i,j is the j-th activity name of
        the i-th trace.
    """
    event_log = pm4py.read_xes(path_to_log_file)

    return [[event[key] for event in trace] for trace in event_log][:limit]


def import_petri_net(path_to_model_file: str):
    """Import a petri net from a .pnml file.

    :param path_to_model_file: A path to the petri net file to be imported
    :type path_to_model_file: str
    :return: A PM4Py petri net, an initial marking and a final marking
    """
    net, initial_marking, final_marking = pm4py.read_petri_net(path_to_model_file)

    return net, initial_marking, final_marking


def generate_playout(net, initial_marking, final_marking, key: str) -> List[List[str]]:
    """Generates a playout of a petri net.

    :param net: A PM4Py petri net
    :param initial_marking: The initial marking of the petri net
    :param final_marking: The final marking of a petri net
    :param key: Activity name key for the given petri net
    :type key: str
    :return: A List[List[str]], where the entry i,j is the j-th activity name
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

    :param dissimilarity_matrix: The NumPy dissimilarity matrix
    :type dissimilarity_matrix: np.ndarray
    """

    def __init__(self, dissimilarity_matrix: np.ndarray):
        self._dissimilarity_matrix = dissimilarity_matrix

    def get_dissimilarity_matrix(self) -> np.ndarray:
        """
        :return: The dissimilarity matrix where an entry at [i,j] is the dissimilarity
            between model trace i and real trace j.
        """
        return self._dissimilarity_matrix

    def calc_fitness(self) -> float:
        """Calculates the fitness value for this dissimilarity matrix.

        :return: The fitness value as a float in [0,1]
        """
        fitness = 1 - np.average(self.get_dissimilarity_matrix().min(axis=0))
        return fitness

    def calc_precision(self) -> float:
        """Calculates the precision value for this dissimilarity matrix.

        :return: The precision value as a float in [0,1]
        """
        precision = 1 - np.average(self.get_dissimilarity_matrix().min(axis=1))
        return precision

    def save(self, path: str):
        """Saves this dissimilarity matrix to a given path via NumPy.

        :param path: The path to save to
        :type path: str
        """
        np.save(path, self.get_dissimilarity_matrix())

    @staticmethod
    def load(path) -> "DissimilarityMatrix":
        """Loads a dissimilarity matrix from a given path via NumPy.

        :param path: The path to load from
        :type path: str
        :return: The dissimilarity matrix object
        """
        matrix = np.load(path)
        return DissimilarityMatrix(matrix)


class EmbeddingConformance(ABC):
    """Abstract base class for conformance checking algorithms based on embeddings.

    The logic of each implementation is defined in :py:func:`_calc_embeddings` and
    :py:func:`_calc_dissimilarity`.

    .. note:: The user should only call :py:func:`execute` and the constructor of some
        implementation. The implementations can be found in the :doc:`algorithms`
        module.
    """

    def execute(
        self, model_traces: List[List[str]], real_traces: List[List[str]]
    ) -> DissimilarityMatrix:
        """Executes this algorithm on the given traces.

        This method is stateless, i.e. no changes are made to `self`.
        It is not static to allow for overriding the abstract methods.

        :param model_traces: The traces coming from the model.
            Rows of the dissimilarity matrix.
        :type model_traces: List[List[str]]
        :param real_traces: The traces coming from the real log.
            Columns of the dissimilarity matrix.
        :type real_traces: List[List[str]]
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
    ) -> Tuple[Union[np.ndarray, List[Any]], Union[np.ndarray, List[Any]], Any]:
        """Calculates the embeddings of the traces.

        .. warning:: This method should not be called directly.
            It is called by :py:func:`execute`.

        :param model_traces: The traces coming from the model.
        :type model_traces: List[List[str]]
        :param real_traces: The traces coming from the real log.
        :type real_traces: List[List[str]]
        :return: The embeddings of the traces of the model and real log
            and an implementation-specific context object.
        """
        pass  # pragma: no cover

    @abstractmethod
    def _calc_dissimilarity(
        self, model_embedding: Any, real_embedding: Any, context: Any
    ) -> float:
        """Calculates the dissimilarity between two embeddings.

        .. warning:: This method should not be called directly.
            It is called by :py:func:`execute`.

        :param model_embedding: The embedding of the model trace
        :type model_embedding: Any
        :param real_embedding: The embedding of the real trace
        :type real_embedding: Any
        :param context: The context object (implementation specific)
        :type context: Any
        :return: A floating-point value in [0, 1] where 1 is the maximum dissimilarity
        """
        pass  # pragma: no cover
