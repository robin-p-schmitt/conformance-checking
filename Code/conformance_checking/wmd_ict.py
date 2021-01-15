from distances import calc_wmd, calc_ict

def main():
    # create some embeddings as example
    # (int, int, ...): int = embedding of a activity: count of this activity within a trace
    model_embedding = {(0.4, 0.3): 3, 
                       (0.2, 0.6): 1,
                       (0.5, 0.9): 2}
    real_embedding = {(0.4, 0.3): 2}

    # calculate WMD between these two traces
    print("WMD: ", calc_wmd(model_embedding, real_embedding))
    print("ICT: ", calc_ict(model_embedding, real_embedding))

if __name__ == "__main__":
    main()
