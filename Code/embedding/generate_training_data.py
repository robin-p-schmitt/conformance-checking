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

# assuming that traces are strings in form of e.g. "abcddedegf", each alphabet representing an activity
# and that log is a collection of such traces
def generate_training_data(log, window_size, num_ns, seed):
	targets, contexts, labels = [], [], []

	# extract all the activities (without duplicates) from traces, and save them in activies
	activities = list()
	for trace in log:
		for activity in trace:  # iterate through activities in trace
			if activity not in activities:
				activities.append(activity)

	# indexing activities
	vocab, index = {}, 1  # start indexing form 1
	vocab['<pad>'] = 0  # add a padding activity
	for activity in activities:
		if activity not in vocab:
			vocab[activity] = index
			index += 1

	vocab_size = len(vocab)

	# create inverse vocabulary to save mapping from integer indicies
	# may not need it
	inverse_vocab = {index: activity for activity, index in vocab.items()}

	for trace in tqdm.tqdm(log):
		# vectorize trace
		vectorized_trace = []
		for activity in trace:
			vectorized_trace.append(vocab[activity])

		# generate positive skip-gram pairs for a trace
		positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(vectorized_trace, vocabulary_size=vocab_size, window_size=window_size, negative_samples=0)

		# create negative sampling for each positive skip-gram pair
		for target_activity, context_activity in positive_skip_grams:
			context_class = tf.expand_dims(tf.constant([context_activity], dtype="int64"), 1)
			negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(true_classes=context_class, num_true=1, num_sampled=num_ns, unique=True, range_max=vocab_size, seed=SEED, name="negative_sampling")

		 	# Build context and label vectors (for one target word)
			negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)

			context = tf.concat([context_class, negative_sampling_candidates], 0)
			label = tf.constant([1] + [0]*num_ns, dtype="int64")

			targets.append(target_activity)
			contexts.append(context)
			labels.append(label)

	return targets, contexts, label

traces = ["aabdcdgecde", "aabdcdgecde", "aabdcdgecde"]
SEED = 42 
AUTOTUNE = tf.data.AUTOTUNE
print(generate_training_data(traces, 3, 4, SEED))







