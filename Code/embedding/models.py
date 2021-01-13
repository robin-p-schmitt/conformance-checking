from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Dot, Embedding, Flatten, GlobalAveragePooling1D, Reshape, Average, Concatenate
import numpy as np

class Act2Vec(Model):
  def __init__(self, vocab_size, embedding_dim, num_ns):
    super(Act2Vec, self).__init__()
    self.target_embedding = Embedding(vocab_size, 
                                      embedding_dim,
                                      input_length=1,
                                      name="w2v_embedding", )
    self.context_embedding = Embedding(vocab_size, 
                                       embedding_dim, 
                                       input_length=num_ns+1)

    self.dots = Dot(axes=(3,2))
    self.flatten = Flatten()

  def call(self, pair):
    target, context = pair
    we = self.target_embedding(target)
    ce = self.context_embedding(context)
    dots = self.dots([ce, we])
    return self.flatten(dots)

class Trace2Vec(Model):
  def __init__(self, trace_vocab_size, act_vocab_size, embedding_dim, context_size):
    super(Trace2Vec, self).__init__()
    self.act_embedding = Embedding(act_vocab_size, 
                                      embedding_dim,
                                      input_length=context_size)
    self.trace_embedding = Embedding(trace_vocab_size, 
                                       embedding_dim, 
                                       input_length=1)
    self.concatenate = Concatenate(axis=1)
    self.average = Average()
    self.flatten = Flatten()
    self.dense = Dense(act_vocab_size, activation="softmax")

  def call(self, pair):
    trace, act_context = pair

    act_emb = self.act_embedding(act_context)
    trace_emb = self.trace_embedding(trace)

    concatenate = self.concatenate([act_emb, trace_emb])
    #average = self.average([*concatenate])
    return self.dense(self.flatten(concatenate))