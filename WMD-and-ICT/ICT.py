from pyemd import emd
import numpy as np
import copy
import collections

def ACT(p, q, C, k): #for now C is new every trace comparison, ADD LATER old used for the early stopping
    t = 0
    for i in range(0, len(p)):
        pi = p[i] #the weight of the ith element in p trace
        if pi == 0.: #if this activity is not actually in p pi will be zero
            continue
        dummy_s = np.argsort(C[i]) #have to change to only use the thing where q[j] != 0
        s = np.ones(k, dtype=int)
        it = 0
        j = 0
        while it<k and j<len(dummy_s):
            if q[dummy_s[j]] != 0.:
                s[it] = int(dummy_s[j])
                it = it + 1
            j = j+1
        l = 0
        while l<k and pi>0:
            r = min(pi, q[s[l]])
            pi = pi - r
            t = t + r*C[i, s[l]] 
            l = l+1
        if pi != 0:
            t =  t + pi*C[i, s[l-1]]
    return t

def _calc_dissimilarity(model_embedding, real_embedding, k):
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

    dist = ACT(d_model, d_real, distance_matrix, k)

    return dist

# test
model_embedding = {(0.1, 0.2): 1,
                   (0.3, 1.0): 2,
                   (-0.5, -0.3): 1}
real_embedding = {(0.1, 0.6): 1,
                  (0.3, 1.3): 2,
                  (-0.5, -0.3): 1}
print(_calc_dissimilarity(model_embedding, real_embedding, 5))