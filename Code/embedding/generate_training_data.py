import numpy as np
import tensorflow as tf
import tqdm


def generate_activity_vocab(log):
    # extract all the activities (without duplicates) from traces,
    # and save them in activies
    vocab, index = {}, 1  # start indexing form 1
    vocab["<pad>"] = 0
    for trace in log:
        for activity in trace:  # iterate through activities in trace
            if activity not in vocab:
                vocab[activity] = index
                index += 1

    return vocab


def generate_trace_vocab(log, act_vocab):
    vocab, index = {}, 0
    for trace in log:
        trace_hash = hash_trace(trace, act_vocab)
        if trace_hash not in vocab:
            vocab[trace_hash] = index
            index += 1

    return vocab


def hash_trace(trace, act_vocab):
    return ".".join([str(index) for index in vectorize_trace(trace, act_vocab)])


def vectorize_trace(trace, vocab):
    vectorized_trace = []
    for activity in trace:
        vectorized_trace.append(vocab[activity])

    return vectorized_trace


def generate_act2vec_training_data(log, vocab, window_size, num_ns):

    targets, contexts, labels = [], [], []
    vocab_size = len(vocab)

    for trace in tqdm.tqdm(log):
        # vectorize trace
        vectorized_trace = vectorize_trace(trace, vocab)

        # generate positive skip-gram pairs for a trace
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            vectorized_trace,
            vocabulary_size=vocab_size,
            window_size=window_size,
            negative_samples=0,
        )

        # create negative sampling for each positive skip-gram pair
        for target_activity, context_activity in positive_skip_grams:
            context_class = tf.expand_dims(
                tf.constant([context_activity], dtype="int64"), 1
            )
            negative_sampling_candidates, _, _ = tf.random.uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=num_ns,
                unique=True,
                range_max=vocab_size,
                name="negative_sampling",
            )

            # Build context and label vectors (for one target word)
            negative_sampling_candidates = tf.expand_dims(
                negative_sampling_candidates, 1
            )

            context = tf.concat([context_class, negative_sampling_candidates], 0)
            label = tf.constant([1] + [0] * num_ns, dtype="int64")

            targets.append(target_activity)
            contexts.append(context)
            labels.append(label)

    return targets, contexts, labels


def generate_trace2vec_training_data(log, vocab_act, vocab_trace, window_size):
    traces, contexts, labels = [], [], []

    act_vocab_size = len(vocab_act)

    for trace in tqdm.tqdm(log):
        vectorized_trace = vectorize_trace(trace, vocab_act)

        for i, activity in enumerate(vectorized_trace):

            label = np.zeros(act_vocab_size)
            label[activity] = 1

            trace_index = vocab_trace[hash_trace(trace, vocab_act)]

            traces.append(trace_index)
            contexts.append(get_context(vectorized_trace, i, window_size))
            labels.append(label)

    return traces, contexts, labels


def get_context(trace, index, window_size):
    left = max(0, index - window_size)
    right = min(len(trace) - 1, index + window_size)
    left_num_padding = window_size - (index - left)
    right_num_padding = window_size - (right - index)
    num_padding = left_num_padding + right_num_padding

    context = trace[left : right + 1] + [0] * num_padding
    context.remove(trace[index])

    return context
