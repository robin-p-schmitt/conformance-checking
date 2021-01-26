from conformance_checking.embedding.embedding_generator import (
    Activity_Embedding_generator,
    Trace_Embedding_generator,
)
from pm4py.objects.log.importer.xes import importer as xes_importer
import matplotlib.pyplot as plt


# load a log with pm4py
variant = xes_importer.Variants.ITERPARSE
parameters = {variant.value.Parameters.MAX_TRACES: 2000}
log = xes_importer.apply(
    "data/BPI_Challenge_2012.xes", variant=variant, parameters=parameters
)

# get log as format List[List[str]]
log = [[event["concept:name"] for event in trace] for trace in log]

# EVALUATE ACT2VEC HYPERPARAMETERS

# effects of different hyperparameters on the accuracy
parameters = {
    "window_size": [],
    "num_ns": [],
    "batch_size": [],
    "embedding_size": [],
}

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

# plot results
for key in parameters:
    plt.figure()
    plt.title("Act2Vec")
    plt.xlabel(key)
    plt.ylabel("accuracy")
    plt.xticks(parameters[key])
    plt.plot(parameters[key], results[key])
    path = "conformance_checking/embedding/hyperparameter_eval/act2vec/"
    plt.savefig(path + "{}_eval.png".format(key))

# EVALUATE TRACE2VEC HYPERPARAMETERS

# effects of different hyperparameters on the accuracy
parameters = {
    "window_size": [],
    "batch_size": [],
    "embedding_size": [],
}

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
    gen = Trace_Embedding_generator(log, embedding_size=embedding_size, auto_train=True)
    acc = gen.evaluate_model()
    parameters["embedding_size"].append(embedding_size)
    results["embedding_size"].append(acc)

# plot results
for key in parameters:
    plt.figure()
    plt.title("Trace2Vec")
    plt.xlabel(key)
    plt.ylabel("accuracy")
    plt.xticks(parameters[key])
    plt.plot(parameters[key], results[key])
    path = "conformance_checking/embedding/hyperparameter_eval/trace2vec/"
    plt.savefig(path + "{}_eval.png".format(key))
