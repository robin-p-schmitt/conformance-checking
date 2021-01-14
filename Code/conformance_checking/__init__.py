from abc import ABC, abstractmethod
from typing import List, Tuple, Any

import numpy as np
import pandas as pd
import pm4py


class ImportData:
    """This class is used to import event logs and process models and provides a method
    for generating a playout of a petry-net.
    """

    def import_xes(self, path_to_log_file):
        """Import an event log, starting and ending activities from a .xes file.
        :param path_to_log_file: a path to the log file to be imported
        :return: a list whos indicies correspont to
        event_log, start_activities, end_activities respectively
        """
        event_log = pm4py.read_xes(path_to_log_file)
        start_activities = pm4py.get_start_activities(event_log)
        end_activities = pm4py.get_end_activities(event_log)

        return [event_log, start_activities, end_activities]

    def import_csv(self, path_to_log_file):
        """Import an event log, starting and ending activities from a .csv file.
        :param path_to_log_file: a path to the log file to be imported
        :return: a list, whose indicies correspont to
        event_log, start_activities, end_activities respectively
        """
        event_log = pd.read_csv(path_to_log_file, sep=";")
        event_log = pm4py.format_dataframe(
            event_log,
            case_id="case_id",
            activity_key="activity",
            timestamp_key="timestamp",
        )
        start_activities = pm4py.get_start_activities(event_log)
        end_activities = pm4py.get_end_activities(event_log)

        return [event_log, start_activities, end_activities]

    def import_petry_net(self, path_to_modell_file):
        """Import a petri net from a .pnml file.
        :param path_to_modell_file: a path to the petri net file to be imported
        :return: a list whose indicies correspond to net, initial_marking, final_marking
         respectively.
        """
        net, initial_marking, final_marking = pm4py.read_petri_net(path_to_modell_file)

        return [net, initial_marking, final_marking]

    def generate_playout(self, net, initial_marking, final_marking):
        """Generate a playout given a petri net, initial_marking and final_marking
        :param net: a petri net
        :param initial_marking: the initial marking of the petri net
        :param final_marking: the final marking of a petri net
        :return: a playout log of the petri net
        """

        playout_log = pm4py.simulation.playout.simulator.apply(
            net, initial_marking, final_marking=final_marking
        )

        return playout_log


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
