from conformance_checking.distances import calc_wmd, calc_ict
import numpy as np


def main():
    # create some embeddings as example
    # (int, int, ...): int =
    # embedding of a activity: count of this activity within a trace
    model_embedding = {0: 3, 1: 1, 2: 2}
    real_embedding = {0: 2}
    context = np.array([[0.4, 0.3], [0.2, 0.6], [0.5, 0.9]])

    # calculate WMD between these two traces
    print("WMD: ", calc_wmd(model_embedding, real_embedding, context))
    print("ICT: ", calc_ict(model_embedding, real_embedding, context))


if __name__ == "__main__":
    main()
