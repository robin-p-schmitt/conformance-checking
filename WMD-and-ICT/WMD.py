from pyemd import emd
import numpy as np
import copy
import collections

def _calc_dissimilarity(model_embedding, real_embedding):
    """calculates the dissimilarity between two embeddings.

    :param model_embedding: the embedding of the model trace
    :param real_embedding: the embedding of the real trace
    :return: a floating-point value in [0, 1] where 1 isl
    the maximum dissimilarity
    """

    whole_embedding = set(list(model_embedding.keys()) + list(real_embedding.keys()))
    
    vocab_len = len(whole_embedding)
    # function: calculate normalized count of activity i within its trace
    def calc_d(embeddings: dict):
        d = np.zeros(vocab_len, dtype=np.double)
        # calculate the length of trace
        trace_len = 0
        for _, value in embeddings:
            trace_len += value
            
        for i, embedding in enumerate(whole_embedding):
            count = embeddings.get(embedding, 0)
            d[i] = count / trace_len
        return d

    d_model = calc_d(model_embedding)
    d_real = calc_d(real_embedding)
    
    # calculate Euclidean distance between embeddings word i and word j
    distance_matrix = np.zeros((vocab_len, vocab_len), dtype=np.double)
    for i, embedding1 in enumerate(whole_embedding):
        for j, embedding2 in enumerate(whole_embedding):
            if distance_matrix[i, j] != 0.0: continue
            distance_matrix[i, j] = distance_matrix[j, i] = \
                np.sqrt(np.sum((np.array(embedding1) - np.array(embedding2))**2))

    dist = emd(d_model, d_real, distance_matrix)

    return dist

# test
model_embedding = {(0.1, 0.2): 1,
                   (0.3, 1.0): 2,
                   (-0.5, -0.3): 1}
real_embedding = {(0.1, 0.6): 1,
                  (0.3, 1.3): 2,
                  (-0.5, -0.3): 1}
print(_calc_dissimilarity(model_embedding, real_embedding))