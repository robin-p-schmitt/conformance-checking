from typing import Tuple
import time
import sys
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import conformance_checking as cc
from conformance_checking.algorithms import (
    Act2VecWmdConformance,
    Act2VecIctConformance,
    Trace2VecCosineConformance,
)

variants = ["alpha", "ilp", "ind0", "ind1", "PN"]
algorithms = ["wmd", "ict", "cosine"]
ids = [5, 6, 7, 8, 9, 10, 11, 12]


class BlockPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stdout = self._original_stderr


def load(variant, id):
    if variant in ["alpha", "ilp", "ind0", "ind1", "PN"]:
        ext = "pnml"
    else:
        ext = "xes"
    path = os.path.join("data", "benchmark", "%s-%d.%s" % (variant, id, ext))
    if ext == "pnml":
        return cc.generate_playout(*cc.import_petri_net(path), "concept:name")
    else:
        return cc.import_xes(path, "concept:name")


def benchmark(log_id, variant, algorithm, rounds):
    with BlockPrint():
        model_log = load(variant, log_id)
        real_log = load("GTlog", log_id)
    algorithm = {
        "wmd": Act2VecWmdConformance(),
        "ict": Act2VecIctConformance(),
        "cosine": Trace2VecCosineConformance(),
    }[algorithm]

    times = []
    matrices = []
    precisions = []
    fitnesses = []
    with BlockPrint():
        for i in range(rounds):
            t0 = time.time()
            matrix = algorithm.execute(model_log, real_log)
            times.append(time.time() - t0)
            matrices.append(matrix.get_dissimilarity_matrix())
            fitnesses.append(matrix.calc_fitness())
            precisions.append(matrix.calc_precision())
    times = np.asarray(times)
    matrices = np.asarray(matrices)
    precisions = np.asarray(precisions)
    fitnesses = np.asarray(fitnesses)
    return times, matrices, precisions, fitnesses


def run_benchmarks(rounds=5):
    try:
        data_loaded = np.load("experiments/benchmark/benchmark.npz", allow_pickle=True)
        data = {}
        for k, v in data_loaded.items():
            data[k] = v
    except IOError:
        data = {
            "done": np.zeros((len(ids), len(variants), len(algorithms)), dtype=np.bool),
            "matrix": np.zeros(
                (len(ids), len(variants), len(algorithms)), dtype=np.object
            ),
            "time": np.zeros(
                (len(ids), len(variants), len(algorithms), rounds), dtype=np.float32
            ),
            "precision": np.zeros(
                (len(ids), len(variants), len(algorithms), rounds), dtype=np.float32
            ),
            "fitness": np.zeros(
                (len(ids), len(variants), len(algorithms), rounds), dtype=np.float32
            ),
        }
    data["done"][7, 0, 0] = 0

    todo = []
    for i in range(len(ids)):
        for j in range(len(variants)):
            for k in range(len(algorithms)):
                if not data["done"][i, j, k]:
                    todo.append((i, j, k))
    random.shuffle(todo)

    for i, j, k in tqdm(todo):
        times, matrices, precisions, fitnesses = benchmark(
            ids[i], variants[j], algorithms[k], rounds
        )
        data["done"][i, j, k] = 1
        data["time"][i, j, k] = times
        data["matrix"][i, j, k] = matrices
        data["precision"][i, j, k] = precisions
        data["fitness"][i, j, k] = fitnesses
        np.savez("experiments/benchmark/benchmark.npz", **data)
    print("done")


def metrics():
    metrics = np.zeros((len(ids), len(variants)), dtype=np.object)

    def act_from_traces(traces):
        activities = set()
        for trace in traces:
            for activity in trace:
                activities.add(activity)
        return activities

    for i in range(len(ids)):
        real_log = [tuple(trace) for trace in load("GTlog", ids[i])]
        for j in range(len(variants)):
            model_log = [tuple(trace) for trace in load(variants[j], ids[i])]
            metrics[i, j] = {
                "num_traces": len(real_log) + len(model_log),
                "num_distinct_traces": len(set(real_log)) + len(set(model_log)),
                "num_bidistinct_traces": len(set(real_log + model_log)),
                "num_distinct_activities": len(act_from_traces(real_log))
                + len(act_from_traces(model_log)),
                "num_bidistinct_activities": len(act_from_traces(real_log + model_log)),
                "avg_trace_length": np.average(
                    np.asarray([len(trace) for trace in real_log + model_log])
                ),
            }

    np.save("experiments/benchmark/metrics.npy", metrics)


def to_prec_fit(args):
    matrix = cc.DissimilarityMatrix(args[4])
    return (
        args[0],
        args[1],
        args[2],
        args[3],
        matrix.calc_precision(),
        matrix.calc_fitness(),
    )


def plot():  # noqa: C901
    benchmark = np.load("experiments/benchmark/benchmark.npz", allow_pickle=True)
    precision = benchmark["precision"]
    fitness = benchmark["fitness"]
    time = benchmark["time"]
    metrics = np.load("experiments/benchmark/metrics.npy", allow_pickle=True)
    rounds = benchmark["matrix"][0, 0, 0].shape[0]

    print(
        "Longest running trial: %s" % str(np.unravel_index(np.argmax(time), time.shape))
    )
    print("Rounds: %d" % rounds)

    def plot_prec_fit(key, name, index):
        fig_axs: Tuple[plt.Figure, Tuple[Tuple[plt.Axes]]] = plt.subplots(
            nrows=2, ncols=5, sharey=True, figsize=(13, 6)
        )
        fig, axs = fig_axs
        fig.suptitle(name)
        for j in range(len(variants)):
            ax_prec: plt.Axes = axs[0][j]
            ax_prec.set_title(variants[j])
            if j == 0:
                ax_prec.set_ylabel("Precision")
            ax_prec.set_xlabel("Logs")
            ax_prec.boxplot(precision[:, j, index, :].T, whis=(0, 100))
            ax_fit = axs[1][j]
            if j == 0:
                ax_fit.set_ylabel("Fitness")
            ax_fit.set_xlabel("Logs")
            ax_fit.boxplot(fitness[:, j, index, :].T, whis=(0, 100))
        fig.savefig("experiments/benchmark/plot_prec_fit_%s.png" % key, dpi=300)

    plot_prec_fit("aw", "act2vec with WMD", 0)
    plot_prec_fit("ai", "act2vec with ICT", 1)
    plot_prec_fit("tc", "trace2vec with cosine", 2)

    def plot_avg_time():
        fig_axs: Tuple[plt.Figure, plt.Axes] = plt.subplots(nrows=1, ncols=1)
        fig, ax = fig_axs
        fig.suptitle("Avg. time per method")
        avg_time = np.average(time, axis=(0, 1, 3))
        ratio_time = avg_time / avg_time.max()
        x = np.arange(3)
        ax.bar(x, ratio_time)
        ax.set_ylabel("relative average time")
        ax.set_xticks(x)
        ax.set_xticklabels(
            ("act2vec with WMD", "act2vec with ICT", "trace2vec with cosine")
        )
        fig.savefig("experiments/benchmark/plot_time.png")

    plot_avg_time()

    def plot_time():
        types = [
            ("num_traces", "#traces"),
            ("num_distinct_traces", "#distinct traces per log"),
            ("num_bidistinct_traces", "#distinct traces total"),
            ("num_distinct_activities", "#distinct activities per log"),
            ("num_bidistinct_activities", "#distinct activities total"),
            ("avg_trace_length", "avg. trace length"),
        ]
        fig_axs: Tuple[plt.Figure, Tuple[Tuple[plt.Axes]]] = plt.subplots(
            nrows=3, ncols=len(types), sharey=True, figsize=(18, 10)
        )
        fig, axs = fig_axs
        fig.suptitle("Time over various complexity metrics")
        num_items = len(ids) * len(variants)
        avg_time = np.average(time, axis=3)
        x_by_metric = np.zeros((len(types), len(algorithms), num_items))
        y_by_metric = np.zeros((len(types), len(algorithms), num_items))
        for i, (key, _) in enumerate(types):
            for j in range(len(algorithms)):
                for k in range(len(ids)):
                    for m in range(len(variants)):
                        x_by_metric[i, j, k * len(variants) + m] = metrics[k, m][key]
                        y_by_metric[i, j, k * len(variants) + m] = avg_time[k, m, j]
        for i, (_, name) in enumerate(types):
            ax_aw: plt.Axes = axs[0][i]
            ax_ai: plt.Axes = axs[1][i]
            ax_tc: plt.Axes = axs[2][i]
            if i == 0:
                ax_aw.set_ylabel("time for act2vec with WMD [s]")
                ax_ai.set_ylabel("time for act2vec with ICT [s]")
                ax_tc.set_ylabel("time for trace2vec with cosine [s]")
            ax_aw.scatter(x_by_metric[i, 0], y_by_metric[i, 0])
            ax_ai.scatter(x_by_metric[i, 1], y_by_metric[i, 1])
            ax_tc.scatter(x_by_metric[i, 2], y_by_metric[i, 2])
            ax_tc.set_xlabel(name)
        fig.savefig("experiments/benchmark/plot_times.png", dpi=300)

    plot_time()

    # plt.show()


if __name__ == "__main__":
    # run_benchmarks()
    # metrics()
    plot()
