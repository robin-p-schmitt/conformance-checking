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

"""
This class is used to generate both activity and trace embeddings from an event log.
When this class is initialized, training for calculating embeddings starts directly.
To get embeddings, please use get_trace_embedding() or get_activity_embedding() to get
corresponding embedding.
"""
class Embedding_generator:
    def __init__(self, log, trace2vec_windows_size=3, act2vec_windows_size=3, num_ns=4):
        self.activity_embedding_generator = Activity_Embedding_generator(
                                            log, act2vec_windows_size, num_ns
                                            )
        self.trace_embedding_generator = Trace_Embedding_generator(
                                            log, trace2vec_windows_size
                                            )

    def get_activity_embedding(self, model_log, real_log):
        return self.activity_embedding_generator.get_activity_embedding(model_log, real_log)

    def get_trace_embedding(self, model_log, real_log):
        return self.trace_embedding_generator.get_trace_embedding(model_log, real_log)

'''
This class is used to generate only activity embeddings from an event log.
'''

class Activity_Embedding_generator:
    def __init__(self, log, act2vec_windows_size=3, num_ns=4):
        # log is expected to be a data type of List[List[str]]

        # create vocabulary for activities
        self.act_vocab = generate_activity_vocab(log)

        # generate training data for act2vec
        self.act2vec_training_data = {}
        (
            self.act2vec_training_data["targets"],
            self.act2vec_training_data["contexts"],
            self.act2vec_training_data["labels"],
        ) = generate_act2vec_training_data(
            log, self.act_vocab, act2vec_windows_size, num_ns
        )

        # generate embeddings
        print("TRAIN ACT2VEC MODEL")
        self.activity_embedding = self.train_model(
            self.act2vec_training_data["targets"],
            self.act2vec_training_data["contexts"],
            self.act2vec_training_data["labels"],
            self.act_vocab,
            num_ns,
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
        buffer_size=10000,
        embedding_dim=128,
    ):
        dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
        dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=False)

        vocab_size = len(vocab)
        act2vec = Act2Vec(vocab_size, embedding_dim, num_ns)
        act2vec.compile(
            optimizer="adam",
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        act2vec.fit(dataset, epochs=10)

        # we need to return embedding!!
        return act2vec.layers[0].get_weights()[0]

    """
    this function returns an embedding matrix of activities
    """

    def get_activity_embedding(self, model_log, real_log):
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

'''
This class is used to generate only trace embeddings from an event log.
'''
class Trace_Embedding_generator:
    def __init__(self, log, trace2vec_windows_size=3):
        # log is expected to be a data type of List[List[str]]

        # create vocabulary for activities and traces
        self.act_vocab = generate_activity_vocab(log)
        self.trace_vocab = generate_trace_vocab(log, self.act_vocab)


        # generate training data for act2vec and trace2vec
        self.trace2vec_training_data = {}
        (
            self.trace2vec_training_data["targets"],
            self.trace2vec_training_data["contexts"],
            self.trace2vec_training_data["labels"],
        ) = generate_trace2vec_training_data(
            log, self.act_vocab, self.trace_vocab, trace2vec_windows_size
        )

        # generate embeddings
        print("TRAIN TRACE2VEC MODEL")
        self.trace_embedding = self.train_model(
            self.trace2vec_training_data["targets"],
            self.trace2vec_training_data["contexts"],
            self.trace2vec_training_data["labels"],
            self.act_vocab,
            self.trace_vocab,
            trace2vec_windows_size,
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
        buffer_size=10000,
        embedding_dim=128,
    ):
        dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
        dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=False)

        trace2vec = Trace2Vec(
            len(trace_vocab), len(act_vocab), embedding_dim, window_size * 2
        )
        trace2vec.compile(
            optimizer="adam",
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )

        trace2vec.fit(dataset, epochs=10)

        return trace2vec.layers[1].get_weights()[0]

    """
    this function returns an embedding matrix of traces
    """

    def get_trace_embedding(self, model_log, real_log):
        model_indices = [
            self.trace_vocab[hash_trace(trace, self.act_vocab)] for trace in model_log
        ]
        real_indices = [
            self.trace_vocab[hash_trace(trace, self.act_vocab)] for trace in real_log
        ]

        model_emb = self.trace_embedding[model_indices]
        real_emb = self.trace_embedding[real_indices]

        return model_emb, real_emb