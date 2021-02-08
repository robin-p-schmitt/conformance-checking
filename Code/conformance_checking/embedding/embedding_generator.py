"""
.. warning:: This module is mainly designed for internal use.
"""

from conformance_checking.embedding.models import Act2Vec, Trace2Vec
from conformance_checking.embedding.generate_training_data import (
    generate_activity_vocab,
    generate_trace_vocab,
    generate_act2vec_training_data,
    generate_trace2vec_training_data,
    vectorize_trace,
    hash_trace,
)

import tensorflow as tf
import numpy as np
from collections import Counter


class ActivityEmbeddingGenerator:
    """This class is used to generate only activity embeddings from an event log.

    :param log: Where the activities and traces for embedding generation come from
    :type log: List[List[str]]
    :param act2vec_windows_size: Window size for act2vec training
    :type act2vec_windows_size: int
    :param num_ns: Number of negative samples for act2vec training
    :type num_ns: int
    :param auto_train: Whether the training for activity embedding starts automatically
        when an instance of this class is created
    :type auto_train: bool
    :param num_epochs: Number of epochs to train
    :type num_epochs: int
    :param batch_size: Size of mini-batches during training
    :type batch_size: int
    :param embedding_size: Length of the generated embeddings
    :type embedding_size: int
    """

    def __init__(
        self,
        log,
        act2vec_windows_size=2,
        num_ns=2,
        auto_train=False,
        num_epochs=10,
        batch_size=1024,
        embedding_size=128,
    ):
        # log is expected to be a data type of List[List[str]]
        self.log = log

        self.num_ns = num_ns
        self.act2vec_windows_size = act2vec_windows_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.act2vec_training_data = {}

        if auto_train:
            # generate embeddings
            self.start_training()
        else:
            # flag that inidicates whether the model is already trained
            self.trained = False

    def start_training(self):
        """
        This function trains the model.
        """
        # create vocabulary for activities
        self.act_vocab = generate_activity_vocab(self.log)

        # generate training data for act2vec
        (
            self.act2vec_training_data["targets"],
            self.act2vec_training_data["contexts"],
            self.act2vec_training_data["labels"],
        ) = generate_act2vec_training_data(
            self.log, self.act_vocab, self.act2vec_windows_size, self.num_ns
        )

        # generate embeddings
        print("TRAIN ACT2VEC MODEL")
        self.activity_embedding = self._train_model(
            self.act2vec_training_data["targets"],
            self.act2vec_training_data["contexts"],
            self.act2vec_training_data["labels"],
            self.act_vocab,
            self.num_ns,
            self.batch_size,
            self.num_epochs,
            self.embedding_size,
        )

    def _train_model(
        self,
        targets,
        contexts,
        labels,
        vocab,
        num_ns,
        batch_size=1024,
        num_epochs=10,
        embedding_dim=128,
        buffer_size=10000,
    ):
        """This function trains an act2vec model and returns an embedding of activities.

        :param targets, contexts, labels: These are results of generating training data
            from an event log
        :param vocab: Activity vocabulary, an uniquely indexed collection of activities
            from an event log, which was used to generate training data
        :param num_ns: Number of desired negative samples for one positive skip-gram
            batch_size, buffer_size, embedding_dim : set as default
        """

        self.trained = True

        self.act2vec_dataset = tf.data.Dataset.from_tensor_slices(
            ((targets, contexts), labels)
        )
        self.act2vec_dataset = self.act2vec_dataset.shuffle(buffer_size).batch(
            batch_size, drop_remainder=False
        )

        vocab_size = len(vocab)
        self.act2vec = Act2Vec(vocab_size, embedding_dim, num_ns)
        self.act2vec.compile(
            optimizer="adam",
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        self.act2vec.fit(self.act2vec_dataset, epochs=num_epochs, verbose=0)

        # we need to return embedding!!
        return self.act2vec.layers[0].get_weights()[0]

    def evaluate_model(self):
        """Returns the accuracy of the trained act2vec model.

        This function evaluates the act2vec model on the dataset that was used for
        training. The training dataset is used because act2vec is an unsupervised
        algorithm and therefore no heldout dataset is used for evaluation.

        :return: The accuracy score of the evaluation
        :rtype: float
        """
        if not self.trained:
            raise ModelNotTrainedError(
                "model for activity embeddings is not trained yet"
            )

        scores = self.act2vec.evaluate(self.act2vec_dataset)

        return scores[1]

    def get_activity_embedding(self, model_log, real_log, norm=False):
        """This function returns the embeddings for the activities.

        :param model_log: The model log of traces.
        :type model_log: List[List[str]]
        :param real_log: The real log of traces.
        :type real_log: List[List[str]]
        :param norm: Wether to normalize embeddings to length 0.5
        :type norm: bool

        :return: The activity frequencies per trace in both model and real log and
           the embedding matrix.
        :rtype: Tuple[Dict[int, int], Dict[int, int], np.ndarray]
        """
        if not self.trained:
            raise ModelNotTrainedError(
                "model for activity embeddings is not trained yet"
            )
        else:
            model_log_indices = [
                vectorize_trace(trace, self.act_vocab) for trace in model_log
            ]
            real_log_indices = [
                vectorize_trace(trace, self.act_vocab) for trace in real_log
            ]

            model_frequency = []
            for trace in model_log_indices:
                c = Counter()
                for act in trace:
                    c.update([act])
                model_frequency.append(c)

            real_frequency = []
            for trace in real_log_indices:
                c = Counter()
                for act in trace:
                    c.update([act])
                real_frequency.append(c)

            embeddings = self.activity_embedding
            if norm:
                # normalize each embedding to have length 0.5
                norms = np.linalg.norm(embeddings, axis=1)
                embeddings = embeddings / (2 * norms[:, None])

            return model_frequency, real_frequency, embeddings


class TraceEmbeddingGenerator:
    """This class is used to generate only trace embeddings from an event log.

    :param log: Where the activities and traces for embedding generation come from
    :type log: List[List[str]]
    :param trace2vec_windows_size: Window size for trace2vec training
    :type trace2vec_windows_size: int
    :param auto_train: Whether the training for trace embedding starts automatically
        when an instance of this class is created
    :type auto_train: bool
    :param num_epochs: Number of epochs to train
    :type num_epochs: int
    :param batch_size: Size of mini-batches during training
    :type batch_size: int
    :param embedding_size: Length of the generated embeddings
    :type embedding_size: int
    """

    def __init__(
        self,
        log,
        trace2vec_windows_size=2,
        auto_train=False,
        num_epochs=10,
        batch_size=1024,
        embedding_size=128,
    ):
        # log is expected to be a data type of List[List[str]]
        self.log = log

        self.trace2vec_windows_size = trace2vec_windows_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.trace2vec_training_data = {}

        if auto_train:
            # generate embeddings
            self.start_training()

        else:
            self.trained = False

    def start_training(self):
        """
        This function trains the model.
        """
        # create vocabulary for activities and traces
        self.act_vocab = generate_activity_vocab(self.log)
        self.trace_vocab = generate_trace_vocab(self.log, self.act_vocab)

        # generate training data for act2vec and trace2vec
        (
            self.trace2vec_training_data["targets"],
            self.trace2vec_training_data["contexts"],
            self.trace2vec_training_data["labels"],
        ) = generate_trace2vec_training_data(
            self.log, self.act_vocab, self.trace_vocab, self.trace2vec_windows_size
        )

        print("TRAIN TRACE2VEC MODEL")
        self.trace_embedding = self._train_model(
            self.trace2vec_training_data["targets"],
            self.trace2vec_training_data["contexts"],
            self.trace2vec_training_data["labels"],
            self.act_vocab,
            self.trace_vocab,
            self.trace2vec_windows_size,
            self.batch_size,
            self.num_epochs,
            self.embedding_size,
        )

    def _train_model(
        self,
        targets,
        contexts,
        labels,
        act_vocab,
        trace_vocab,
        window_size,
        batch_size=1024,
        num_epochs=10,
        embedding_dim=128,
        buffer_size=10000,
    ):
        """This function trains an trace2vec model and returns an embedding of traces.

        :param targets, contexts, labels: These are results of generating training data
            from an event log
        :param act_vocab: Activity vocabulary, an uniquely indexed collection of
            activities from an event log, which was used to generate training data
        :param trace_vocab: Trace vocabulary, an uniquely indexed collection of traces
            from an event log, which was used to generate training data
        :param window_size: Window size needed for computing context size
        :param batch_size, buffer_size, embedding_dim: Set as default
        """
        self.trained = True

        self.trace2vec_dataset = tf.data.Dataset.from_tensor_slices(
            ((targets, contexts), labels)
        )
        self.trace2vec_dataset = self.trace2vec_dataset.shuffle(buffer_size).batch(
            batch_size, drop_remainder=False
        )

        self.trace2vec = Trace2Vec(
            len(trace_vocab), len(act_vocab), embedding_dim, window_size * 2
        )
        self.trace2vec.compile(
            optimizer="adam",
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )

        self.trace2vec.fit(self.trace2vec_dataset, epochs=num_epochs, verbose=0)

        return self.trace2vec.layers[1].get_weights()[0]

    def evaluate_model(self):
        """Returns the accuracy of the trained trace2vec model.

        This function evaluates the trace2vec model on the dataset that
        was used for training. The training dataset is used because trace2vec
        is an unsupervised algorithm and therefore no heldout dataset is used
        for evaluation.

        :return: accuracy score of the evaluation
        :rtype: float
        """

        if not self.trained:
            raise ModelNotTrainedError("model for trace embeddings is not trained yet")

        scores = self.trace2vec.evaluate(self.trace2vec_dataset)

        return scores[1]

    def get_trace_embedding(self, model_log, real_log, norm=False):
        """This function returns the embeddings for the traces.

        :param model_log: The model log of traces.
        :type model_log: List[List[str]]
        :param real_log: The real log of traces.
        :type real_log: List[List[str]]
        :param norm: Wether to normalize embeddings to length 0.5
        :type norm: bool

        :return: The embeddings of the traces in the model and real log.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        if not self.trained:
            raise ModelNotTrainedError("model for trace embeddings is not trained yet")
        else:
            model_indices = [
                self.trace_vocab[hash_trace(trace, self.act_vocab)]
                for trace in model_log
            ]
            real_indices = [
                self.trace_vocab[hash_trace(trace, self.act_vocab)]
                for trace in real_log
            ]

            embeddings = self.trace_embedding
            if norm:
                # normalize each embedding to have length 0.5
                norms = np.linalg.norm(embeddings, axis=1)
                embeddings = embeddings / (2 * norms[:, None])

            model_emb = embeddings[model_indices]
            real_emb = embeddings[real_indices]

            return model_emb, real_emb


class ModelNotTrainedError(Exception):
    """
    Thrown if a model is used prior to training.
    """

    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return "ModelNotTrainedError, {0}".format(self.message)
        else:
            return "ModelNotTrainedError has been raised"
