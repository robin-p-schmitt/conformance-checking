from conformance_checking.embedding.embedding_generator import (
    Activity_Embedding_generator,
    Trace_Embedding_generator,
)
from pm4py.objects.log.importer.xes import importer as xes_importer
import matplotlib.pyplot as plt
import os


def load_data():
    """Load and return example data.

    Returns:
        log (List[List[str]]): example log.

    """

    # file path
    absPath = os.path.abspath(__file__)
    fileDir = os.path.dirname(absPath)
    conformance = os.path.dirname(fileDir)
    code = os.path.dirname(conformance)
    data = os.path.join(code, "data")

    # load the first 2000 traces of the example log
    variant = xes_importer.Variants.ITERPARSE
    parameters = {variant.value.Parameters.MAX_TRACES: 2000}
    log = xes_importer.apply(
        os.path.join(data, "BPI_Challenge_2012.xes"),
        variant=variant,
        parameters=parameters,
    )

    # get log as format List[List[str]]
    return [[event["concept:name"] for event in trace] for trace in log]


def eval_act2vec(log):
    """Evaluate different parameter sets for act2vec.

    This function starts with a baseline act2vec model and varies
    different parameters in isolation to check how they affect the
    resulting accuracy of the model. The parameters under investigation
    are: window size, number of negative samples, batch size and embedding size.

    Args:
        log (List[List[str]]): log to train the models on.

    Returns:
        parameters (Dict[(str, List)]): parameter values.
        results (Dict[(str, List)]): resulting accuracies.

    """

    # different parameter values
    parameters = {
        "window_size": [],
        "num_ns": [],
        "batch_size": [],
        "embedding_size": [],
    }

    # corresponding accuracies
    results = {
        "window_size": [],
        "num_ns": [],
        "batch_size": [],
        "embedding_size": [],
    }

    # different window sizes
    for window_size in [1, 2, 3, 4, 5]:
        gen = Activity_Embedding_generator(log, window_size, auto_train=True)
        acc = gen.evaluate_model()
        parameters["window_size"].append(window_size)
        results["window_size"].append(acc)

    # different number of negative samples
    for num_ns in [1, 2, 3, 4, 5]:
        gen = Activity_Embedding_generator(log, num_ns=num_ns, auto_train=True)
        acc = gen.evaluate_model()
        parameters["num_ns"].append(num_ns)
        results["num_ns"].append(acc)

    # different batch sizes
    for batch_size in [64, 128, 512, 1024]:
        gen = Activity_Embedding_generator(log, batch_size=batch_size, auto_train=True)
        acc = gen.evaluate_model()
        parameters["batch_size"].append(batch_size)
        results["batch_size"].append(acc)

    # different embedding sizes
    for embedding_size in [16, 32, 64, 128]:
        gen = Activity_Embedding_generator(
            log, embedding_size=embedding_size, auto_train=True
        )
        acc = gen.evaluate_model()
        parameters["embedding_size"].append(embedding_size)
        results["embedding_size"].append(acc)

    return parameters, results


def eval_trace2vec(log):
    """Evaluate different parameter sets for trace2vec.

    This function starts with a baseline trace2vec model and varies
    different parameters in isolation to check how they affect the
    resulting accuracy of the model. The parameters under investigation
    are: window size, batch size and embedding size.

    Args:
        log (List[List[str]]): log to train the models on.

    Returns:
        parameters (Dict[(str, List)]): parameter values.
        results (Dict[(str, List)]): resulting accuracies.

    """

    # parameter values
    parameters = {
        "window_size": [],
        "batch_size": [],
        "embedding_size": [],
    }

    # resulting accuracies
    results = {
        "window_size": [],
        "batch_size": [],
        "embedding_size": [],
    }

    # different window sizes
    for window_size in [1, 2, 3, 4, 5, 8]:
        gen = Trace_Embedding_generator(log, window_size, auto_train=True)
        acc = gen.evaluate_model()
        parameters["window_size"].append(window_size)
        results["window_size"].append(acc)

    # different batch sizes
    for batch_size in [64, 128, 512, 1024]:
        gen = Trace_Embedding_generator(log, batch_size=batch_size, auto_train=True)
        acc = gen.evaluate_model()
        parameters["batch_size"].append(batch_size)
        results["batch_size"].append(acc)

    # different embedding sizes
    for embedding_size in [16, 32, 64, 128]:
        gen = Trace_Embedding_generator(
            log, embedding_size=embedding_size, auto_train=True
        )
        acc = gen.evaluate_model()
        parameters["embedding_size"].append(embedding_size)
        results["embedding_size"].append(acc)

    return parameters, results


def save_results(parameters, results, model):
    """Save plots of parameter values against resulting accuracies.

    Args:
        parameters (Dict[(str, List)]): parameter values.
        results (Dict[(str, List)]): resulting accuracies.
        model (str): name of model (used to name destination folder)
    """
    for key in parameters:
        plt.figure()
        plt.title(model)
        plt.xlabel(key)
        plt.ylabel("accuracy")
        plt.xticks(parameters[key])
        plt.plot(parameters[key], results[key])
        path = "conformance_checking/embedding/hyperparameter_eval/"
        plt.savefig(path + "{}/{}_eval.png".format(model, key))


if __name__ == "__main__":
    log = load_data()

    # get different parameter sets together with corresponding accuracies
    parameters, results = eval_act2vec(log)
    # save the resulting plots as png
    save_results(parameters, results, "act2vec")

    # get different parameter sets together with corresponding accuracies
    parameters, results = eval_trace2vec(log)
    # save the resulting plots as png
    save_results(parameters, results, "trace2vec")
