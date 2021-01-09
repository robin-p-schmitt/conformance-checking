#this file is only for testing purposes.

import io
import itertools
import numpy as np
import os
import re
import string
import tensorflow as tf
import tqdm

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Dot, Embedding, Flatten, GlobalAveragePooling1D, Reshape
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

#https://www.tensorflow.org/tutorials/text/word2vec

# extract all the activities (without duplicates) from traces, and save them in tokens
traces = ["aabdcdgecde", "aabdcdgecde", "aabdcdgecde"] # trace as string 
tokens = list()
for trace in traces:
	for activity in trace: #iterate through activities in trace
		if activity not in tokens:
			tokens.append(activity)

#indexing activities
vocab, index = {} , 1 # start indexing form 1
vocab['<pad>'] = 0 # add a padding token
for token in tokens:
	if token not in vocab:
		vocab[token] = index
		index += 1

vocab_size = len(vocab)
#print(vocab)

# create inverse vocabulary to save mapping from integer indicies
inverse_vocab = {index: token for token, index in vocab.items()}
#print(inverse_vocab)

# vectorize traces
vectorized_traces = list()
for trace in traces:
	vectorized_trace = []
	for activity in trace:
		vectorized_trace.append(vocab[activity])
	vectorized_traces.append(vectorized_trace)
#print(vectorized_traces)

# create positive skip grams
window_size = 3 # according to paper
positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
      vectorized_traces[0], 
      vocabulary_size=vocab_size,
      window_size=window_size,
      negative_samples=0)
print(len(positive_skip_grams))
for skipgram in positive_skip_grams:
	print(skipgram, '\n')



