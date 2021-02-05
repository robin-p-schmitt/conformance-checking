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
from collections import Counter

class ActivityEmbeddingGenerator:
    """
    This class is used to generate only activity embeddings from an event log.

    @param      log: where the activities and traces for embedding generation come from
    @param      act2vec_windows_size: window_size for act2vec training
    @param      num_ns: number of negative samples for act2vec training
    @param      auto_train: whether the training for activity embedding
                            starts automatically when an instance of this class is created
    @param      num_epochs: the number of iterations through the whole dataset when
                            training the embedding model
    @param      batch_size: the size of the batches used for gradient descent while
                            training the embedding model
    @param      embedding_size: the size of the embeddings that activities are represented as
    @param      training_verbose: the amount of output during model training:
                                    0: no output
                                    1: progress bar per epoch
                                    2: one line per epoch + loss and accuracy
                                    3: one line per epoch
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
        training_verbose = 1,
    ):
        # log is expected to be a data type of List[List[str]]
        self.log = log

        self.num_ns = num_ns
        self.act2vec_windows_size = act2vec_windows_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.act2vec_training_data = {}

        if training_verbose < 0 or training_verbose > 3:
            raise ValueError("parameter 'training_verbose' needs to be between 0 and 3")
        self.training_verbose = training_verbose

        if auto_train:
            # generate embeddings
            self.start_training()
        else:
            # flag that inidicates whether the model is already trained
            self.trained = False

    """
    this function starts to train the model
    """

    def start_training(self):
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
        self.activity_embedding = self.train_model(
            self.act2vec_training_data["targets"],
            self.act2vec_training_data["contexts"],
            self.act2vec_training_data["labels"],
            self.act_vocab,
            self.num_ns,
            self.batch_size,
            self.num_epochs,
            self.embedding_size,
            verbose = self.training_verbose
        )

    """
    this function trains an act2vec model and returns an embedding of activities
    @param  targets, contexts, labels: these are results of generating training data
            from an event log
            vocab: activity vocabulary, an uniquely indexed collection of activities
            from an event log, which was used to generate training data
            num_ns: number of desired negative samples for one positive skip-gram
            batch_size, buffer_size, embedding_dim : set as default
    """

    def train_model(
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
        verbose = 1,
    ):
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

        self.act2vec.fit(self.act2vec_dataset, epochs=num_epochs, verbose=verbose)

        # we need to return embedding!!
        return self.act2vec.layers[0].get_weights()[0]

    def evaluate_model(self):
        """Returns the accuracy of the trained act2vec model.

        This function evaluates the act2vec model on the dataset that was used for
        training. The training dataset is used because act2vec is an unsupervised
        algorithm and therefore no heldout dataset is used for evaluation.

        Returns:
            score (float): accuracy score of the evaluation.

        """
        if not self.trained:
            raise ModelNotTrainedError(
                "model for activity embeddings is not trained yet"
            )

        scores = self.act2vec.evaluate(self.act2vec_dataset)

        return scores[1]

    """
    this function returns an embedding matrix of activities
    """

    def get_activity_embedding(self, model_log, real_log):
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

            return model_frequency, real_frequency, self.activity_embedding

class TraceEmbeddingGenerator:
    """
    This class is used to generate only trace embeddings from an event log.

    @param      log: where the activities and traces for embedding generation come from
    @param      trace2vec_windows_size: window_size for trace2vec training
    @param      auto_train: whether the training for trace embedding starts
                            automatically when an instance of this class is created
    @param      num_epochs: the number of iterations through the whole dataset when
                            training the embedding model
    @param      batch_size: the size of the batches used for gradient descent while
                            training the embedding model
    @param      embedding_size: the size of the embeddings that activities are represented as
    @param      training_verbose: the amount of output during model training:
                                    0: no output
                                    1: progress bar per epoch
                                    2: one line per epoch + loss and accuracy
                                    3: one line per epoch
    """
    def __init__(
        self,
        log,
        trace2vec_windows_size=2,
        auto_train=False,
        num_epochs=10,
        batch_size=1024,
        embedding_size=128,
        training_verbose = 1
    ):
        # log is expected to be a data type of List[List[str]]
        self.log = log

        self.trace2vec_windows_size = trace2vec_windows_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.trace2vec_training_data = {}

        if training_verbose < 0 or training_verbose > 3:
            raise ValueError("parameter 'training_verbose' needs to be between 0 and 3")
        self.training_verbose = training_verbose

        if auto_train:
            # generate embeddings
            self.start_training()

        else:
            self.trained = False

    """
    this function starts to train model
    """

    def start_training(self):
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
        self.trace_embedding = self.train_model(
            self.trace2vec_training_data["targets"],
            self.trace2vec_training_data["contexts"],
            self.trace2vec_training_data["labels"],
            self.act_vocab,
            self.trace_vocab,
            self.trace2vec_windows_size,
            self.batch_size,
            self.num_epochs,
            self.embedding_size,
            verbose = self.training_verbose
        )

    """
    this function trains an trace2vec model and returns an embedding of traces
    @param  targets, contexts, labels: these are results of generating training data
            from an event log
            act_vocab: activity vocabulary, an uniquely indexed collection of
            activities from an event log, which was used to generate training data
            trace_vocab: trace vocabulary, an uniquely indexed collection of traces
            from an event log, which was used to generate training data
            window_size: window size needed for computing context size
            batch_size, buffer_size, embedding_dim : set as default
    """

    def train_model(
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
        verbose = 1
    ):
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

        self.trace2vec.fit(self.trace2vec_dataset, epochs=num_epochs, verbose=verbose)

        return self.trace2vec.layers[1].get_weights()[0]

    def evaluate_model(self):
        """Returns the accuracy of the trained trace2vec model.

        This function evaluates the trace2vec model on the dataset that
        was used for training. The training dataset is used because trace2vec
        is an unsupervised algorithm and therefore no heldout dataset is used
        for evaluation.

        Returns:
            score (float): accuracy score of the evaluation.

        """

        if not self.trained:
            raise ModelNotTrainedError("model for trace embeddings is not trained yet")

        scores = self.trace2vec.evaluate(self.trace2vec_dataset)

        return scores[1]

    """
    this function returns an embedding matrix of traces
    """

    def get_trace_embedding(self, model_log, real_log):
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

            model_emb = self.trace_embedding[model_indices]
            real_emb = self.trace_embedding[real_indices]

            return model_emb, real_emb


class ModelNotTrainedError(Exception):
    """
    custom error class for ModelNotTrainedError
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
