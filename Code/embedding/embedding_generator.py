from generate_training_data import *
from models import *

import numpy as np
import tensorflow as tf
import tqdm

class Embedding_generator:
    def __init__(self, log, trace2vec_windows_size=3, act2vec_windows_size=3, num_ns=4):
        # log is expected to be a data type of List[List[str]]

        # create vocabulary for activities and traces
        self.act_vocab = generate_activity_vocab(log)
        self.trace_vocab = generate_trace_vocab(log, self.act_vocab)

        # generate training datas for act2vec and trace2vec
        self.act2vec_training_data = {}
        self.trace2vec_training_data = {}
        self.act2vec_training_data["targets"], self.act2vec_training_data["contexts"], self.act2vec_training_data["labels"] = generate_act2vec_training_data(traces, act_vocab, window_size, num_ns)
        self.trace2vec_training_data["targets"], self.trace2vec_training_data["contexts"], self.trace2vec_training_data["labels"] = generate_trace2vec_training_data(traces, act_vocab, trace_vocab, window_size)

        # generate embeddings
        activity_embedding = train_act2vec_model(1024, 10000, 128, self.act2vec_training_data["targets"], self.act2vec_training_data["contexts"], self.act2vec_training_data["labels"], num_ns)
        trace_embedding = train_trace2vec_model(1024, 10000, 128, self.trace2vec_training_data["targets"], self.trace2vec_training_data["contexts"], self.trace2vec_training_data["labels"], act_vocab, trace_vocab, window_size)

    def train_act2vec_model(self, targets, contexts, labels, vocab, num_ns, batch_size=1024, buffer_size=10000, embedding_dim=128):
        dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
        dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=False)

        vocab_size = len(vocab)
        act2vec = Act2Vec(vocab_size, embedding_dim, num_ns)
        act2vec.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

        act2vec.fit(dataset, epochs=50)

        # we need to return embedding!!
        return act2vec.get_target_embedding()

    def train_trace2vec_model(self, targets, contexts, labels, act_vocab, trace_vocab, window_size, batch_size=1024, buffer_size=10000, embedding_dim=128):
        dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
        dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=False)

        trace2vec = Trace2Vec(len(trace_vocab), len(act_vocab), embedding_dim, window_size * 2)
        trace2vec.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

        trace2vec.fit(dataset, epochs=10)

        return trace2vec.get_trace_embedding()

    def get_activity_embedding(self):
        return self.activity_embedding

    def get_trace_embedding(self):
        return self,trace_embedding
