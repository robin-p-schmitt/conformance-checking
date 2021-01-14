from generate_training_data import *
from models import *

import numpy as np
import tensorflow as tf
import tqdm

'''
This class is used to generate embeddings.
When this class is initiaized, training for calculating embeddings starts directly.
To get embeddings, please use get_trace_embedding() or get_activity_embedding() to get correspoding embedding
'''
class Embedding_generator:
    def __init__(self, log, trace2vec_windows_size=3, act2vec_windows_size=3, num_ns=4):
        # log is expected to be a data type of List[List[str]]

        # this log is just for testing the functionality of the class
        log = xes_importer.apply('logs/BPI_Challenge_2012.xes')
        log = [[event["concept:name"] for event in trace] for trace in log][:2000]

        # create vocabulary for activities and traces
        self.act_vocab = generate_activity_vocab(log)
        self.trace_vocab = generate_trace_vocab(log, self.act_vocab)

        # generate training data for act2vec and trace2vec
        self.act2vec_training_data = {}
        self.trace2vec_training_data = {}
        self.act2vec_training_data["targets"], self.act2vec_training_data["contexts"], self.act2vec_training_data["labels"] = generate_act2vec_training_data(log, self.act_vocab, act2vec_windows_size, num_ns)
        self.trace2vec_training_data["targets"], self.trace2vec_training_data["contexts"], self.trace2vec_training_data["labels"] = generate_trace2vec_training_data(log, self.act_vocab, self.trace_vocab, trace2vec_windows_size)

        # generate embeddings
        self.activity_embedding = self.train_act2vec_model(self.act2vec_training_data["targets"], self.act2vec_training_data["contexts"], self.act2vec_training_data["labels"], self.act_vocab, num_ns)
        self.trace_embedding = self.train_trace2vec_model(self.trace2vec_training_data["targets"], self.trace2vec_training_data["contexts"], self.trace2vec_training_data["labels"], self.act_vocab, self.trace_vocab, trace2vec_windows_size)

    '''
    this function trains an act2vec model and returns an embedding of activities
    @param  targets, contexts, labels: these are results of generating training data from an event log
            vocab: activity vocabulary, an uniquely indexed collection of activities from an event log, which was used to generate training data
            num_ns: number of desired negative samples for one positive skip-gram
            batch_size, buffer_size, embedding_dim : set as default
    '''
    def train_act2vec_model(self, targets, contexts, labels, vocab, num_ns, batch_size=1024, buffer_size=10000, embedding_dim=128):
        dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
        dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=False)

        vocab_size = len(vocab)
        act2vec = Act2Vec(vocab_size, embedding_dim, num_ns)
        act2vec.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

        act2vec.fit(dataset, epochs=50)

        # we need to return embedding!!
        return act2vec.get_target_embedding()

    '''
    this function trains an trace2vec model and returns an embedding of traces
    @param  targets, contexts, labels: these are results of generating training data from an event log
            act_vocab: activity vocabulary, an uniquely indexed collection of activities from an event log, which was used to generate training data
            trace_vocab: trace vocabulary, an uniquely indexed collection of traces from an event log, which was used to generate training data
            window_size: window size needed for computing context size
            batch_size, buffer_size, embedding_dim : set as default
    '''
    def train_trace2vec_model(self, targets, contexts, labels, act_vocab, trace_vocab, window_size, batch_size=1024, buffer_size=10000, embedding_dim=128):
        dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
        dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=False)

        trace2vec = Trace2Vec(len(trace_vocab), len(act_vocab), embedding_dim, window_size * 2)
        trace2vec.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

        trace2vec.fit(dataset, epochs=10)

        return trace2vec.get_trace_embedding()

    '''
    this function returns an embedding matrix of activities
    '''
    def get_activity_embedding(self):
        return self.activity_embedding

    '''
    this function returns an embedding matrix of traces
    '''
    def get_trace_embedding(self):
        return self.trace_embedding
