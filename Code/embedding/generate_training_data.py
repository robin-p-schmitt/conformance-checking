import numpy as np
import tensorflow as tf
import tqdm

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Dot, Embedding, Flatten, GlobalAveragePooling1D, Reshape

from models import Act2Vec, Trace2Vec

def generate_activity_vocab(log):
	# extract all the activities (without duplicates) from traces, and save them in activies
	vocab, index = {}, 0  # start indexing form 1
	for trace in log:
		for activity in trace:  # iterate through activities in trace
			if activity not in vocab:
				vocab[activity] = index
				index += 1

	return vocab

def generate_trace_vocab(log):
	vocab, index = {}, 0
	for trace in log:
		if trace not in vocab:
			vocab[trace] = index
			index += 1

	return vocab

def vectorize_trace(trace, vocab):
	vectorized_trace = []
	for activity in trace:
		vectorized_trace.append(vocab[activity])

	return vectorized_trace


# assuming that traces are strings in form of e.g. "abcddedegf", each alphabet representing an activity
# and that log is a collection of such traces
def generate_act2vec_training_data(log, vocab, window_size, num_ns):
	targets, contexts, labels = [], [], []

	# # extract all the activities (without duplicates) from traces, and save them in activies
	# activities = list()
	# for trace in log:
	# 	for activity in trace:  # iterate through activities in trace
	# 		if activity not in activities:
	# 			activities.append(activity)

	# # indexing activities
	# vocab, index = {}, 0  # start indexing form 1
	# #vocab['<pad>'] = 0  # add a padding activity
	# for activity in activities:
	# 	if activity not in vocab:
	# 		vocab[activity] = index
	# 		index += 1

	vocab_size = len(vocab)

	# create inverse vocabulary to save mapping from integer indicies
	# may not need it
	# inverse_vocab = {index: activity for activity, index in vocab.items()}

	for trace in tqdm.tqdm(log):
		# vectorize trace
		vectorized_trace = vectorize_trace(trace, vocab)

		# generate positive skip-gram pairs for a trace
		positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(vectorized_trace, vocabulary_size=vocab_size, window_size=window_size, negative_samples=0)

		# create negative sampling for each positive skip-gram pair
		for target_activity, context_activity in positive_skip_grams:
			context_class = tf.expand_dims(tf.constant([context_activity], dtype="int64"), 1)
			negative_sampling_candidates, _, _ = tf.random.uniform_candidate_sampler(true_classes=context_class, num_true=1, num_sampled=num_ns, unique=True, range_max=vocab_size, name="negative_sampling")

		 	# Build context and label vectors (for one target word)
			negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)

			context = tf.concat([context_class, negative_sampling_candidates], 0)
			label = tf.constant([1] + [0]*num_ns, dtype="int64")

			targets.append(target_activity)
			contexts.append(context)
			labels.append(label)

	return targets, contexts, labels

def generate_trace2vec_training_data(log, vocab_act, vocab_trace, window_size):
	traces, contexts, labels = [], [], []

	act_vocab_size = len(vocab_act)

	for trace in tqdm.tqdm(log):
		vectorized_trace = vectorize_trace(trace, vocab_act)

		#positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(vectorized_trace, vocabulary_size=vocab_size, window_size=window_size, negative_samples=0)

		for i, activity in enumerate(trace):
			
			label = np.zeros(act_vocab_size)
			label[vocab_act[activity]] = 1
			
			trace_index = vocab_trace[trace]

			traces.append(trace_index)
			contexts.append(get_context(trace, i, window_size, vocab_act))
			labels.append(label)

	return traces, contexts, labels		

def get_context(trace, index, window_size, vocab_act):
	left = max(0, index - window_size)
	right = min(len(trace) - 1, index + window_size)
	left_num_padding = window_size - (index - left)
	right_num_padding = window_size - (right - index)
	num_padding = left_num_padding + right_num_padding

	vectorized_trace = vectorize_trace(trace, vocab_act)
	context = vectorized_trace[left:right+1] + [0] * num_padding
	context.remove(index)
	print(context)

	return context


traces = ["abcdefghij"] * 512
vocab = generate_activity_vocab(traces)
num_ns = 4
window_size = 3
'''
targets, contexts, labels = generate_act2vec_training_data(traces, vocab, window_size, num_ns)

BATCH_SIZE = 1024
BUFFER_SIZE = 10000
dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=False)

print(dataset)

embedding_dim = 128
vocab_size = len(vocab)
act2vec = Act2Vec(vocab_size, embedding_dim, num_ns)
act2vec.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#act2vec.fit(dataset, epochs=50)
'''
vocab_trace = generate_trace_vocab(traces)
print(generate_trace2vec_training_data(traces, vocab, vocab_trace, 2))










