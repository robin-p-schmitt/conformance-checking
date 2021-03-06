from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    Dot,
    Embedding,
    Flatten,
    Concatenate,
)


class Act2Vec(Model):
    def __init__(self, vocab_size, embedding_dim, num_ns):
        super(Act2Vec, self).__init__()
        self.target_embedding = Embedding(
            vocab_size,
            embedding_dim,
            input_length=1,
            name="w2v_embedding",
        )
        self.context_embedding = Embedding(
            vocab_size, embedding_dim, input_length=num_ns + 1
        )

        self.dots = Dot(axes=(3, 2))
        self.flatten = Flatten()

    # this method is traced by tensorflow, but not detected by coverage report
    def call(self, pair):  # pragma: nocover
        target, context = pair
        we = self.target_embedding(target)
        ce = self.context_embedding(context)
        dots = self.dots([ce, we])
        return self.flatten(dots)


class Trace2Vec(Model):
    def __init__(self, trace_vocab_size, act_vocab_size, embedding_dim, context_size):
        super(Trace2Vec, self).__init__()
        self.act_embedding = Embedding(
            act_vocab_size, embedding_dim, input_length=context_size
        )
        self.trace_embedding = Embedding(
            trace_vocab_size, embedding_dim, input_length=1
        )
        self.concatenate = Concatenate(axis=1)
        self.flatten = Flatten()
        self.dense = Dense(act_vocab_size, activation="softmax")

    # this method is traced by tensorflow, but not detected by coverage report
    def call(self, pair):  # pragma: nocover
        trace, act_context = pair

        act_emb = self.act_embedding(act_context)
        trace_emb = self.trace_embedding(trace)

        concatenate = self.concatenate([act_emb, trace_emb])
        return self.dense(self.flatten(concatenate))
