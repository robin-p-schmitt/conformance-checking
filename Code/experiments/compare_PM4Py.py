import os
import pm4py
import pandas as pd
import matplotlib.pyplot as plt
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.evaluation.replay_fitness import evaluator as replay_fitness_evaluator
from pm4py.evaluation.precision import evaluator as precision_evaluator
from pm4py.statistics.variants.log import get as variants_module
from pm4py.simulation.playout import simulator
from pm4py.evaluation.earth_mover_distance import evaluator

# Global variable
variants = ["alpha", "ilp", "ind0", "ind1", "PN"]


def import_models(variant):
    net, im, fm = [], [], []
    for ind in range(5, 13):
        file_name = variant + "-" + str(ind) + ".pnml"
        net_curr, im_curr, fm_curr = pm4py.read_petri_net(
            os.path.join(data_path, file_name)
        )
        net.append(net_curr)
        im.append(im_curr)
        fm.append(fm_curr)
    return net, im, fm


def plot(df, title, photo_name, ylabel):
    fig, ax = plt.subplots(1, len(variants), figsize=(13, 4), sharey=True)
    for ind, variant in enumerate(variants):
        if ind == 0:
            ax[ind].set_ylabel(ylabel)
        ax[ind].set_xlabel("Logs")
        ax[ind].bar(df.index.astype(str), df[variant], width=0.6)
        ax[ind].set_title(variant)
    plt.suptitle(title)
    plt.savefig(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "compare_PM4Py/" + photo_name
        )
    )


def token_based_replay(real_logs, petri_nets):
    dic = {}
    for ind, variant in enumerate(variants):
        net, im, fm = petri_nets[ind]
        fitness = []
        for i in range(8):
            fitness.append(
                replay_fitness_evaluator.apply(
                    real_logs[i],
                    net[i],
                    im[i],
                    fm[i],
                    variant=replay_fitness_evaluator.Variants.TOKEN_BASED,
                )["log_fitness"]
            )
        dic[variant] = fitness
    plot(
        pd.DataFrame(dic, index=list(range(1, 9))),
        "Fitness using Token-Based Replay",
        "token_based_replay.png",
        ylabel="Fitness",
    )


def et_conformance(real_logs, petri_nets):
    dic = {}
    for ind, variant in enumerate(variants):
        net, im, fm = petri_nets[ind]
        precision = []
        for i in range(8):
            precision.append(
                precision_evaluator.apply(
                    real_logs[i],
                    net[i],
                    im[i],
                    fm[i],
                    variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN,
                )
            )
        dic[variant] = precision
    plot(
        pd.DataFrame(dic, index=list(range(1, 9))),
        "Precision using ETConformance method",
        "et_conformance.png",
        ylabel="Precision",
    )


def earth_mover_distance(real_logs, petri_nets):
    language = []
    dic = {}
    for i in range(8):
        language.append(variants_module.get_language(real_logs[i]))
    for i, variant in enumerate(variants):
        net, im, fm = petri_nets[i]
        emd = []
        for j in range(8):
            playout_log = simulator.apply(
                net[j],
                im[j],
                fm[j],
            )
            playout_log = simulator.apply(
                net[j],
                im[j],
                fm[j],
                parameters={
                    simulator.Variants.STOCHASTIC_PLAYOUT.value.Parameters.LOG: playout_log  # noqa: E501
                },
                variant=simulator.Variants.STOCHASTIC_PLAYOUT,
            )
            model_language = variants_module.get_language(playout_log)
            emd.append(evaluator.apply(model_language, language[j]))
        dic[variant] = emd
    plot(
        pd.DataFrame(dic, index=list(range(1, 9))),
        "Earth Mover Distance",
        "earth_mover_distance.png",
        ylabel="emd",
    )


if __name__ == "__main__":
    # get directory
    absPath = os.path.abspath(__file__)
    experiment = os.path.dirname(absPath)
    code = os.path.dirname(experiment)
    data_path = os.path.join(code, "data/benchmark")

    # import real logs
    real_logs = []
    variant = xes_importer.Variants.ITERPARSE
    for i in range(5, 13):
        file = "GTlog-" + str(i) + ".xes"
        real_logs.append(
            xes_importer.apply(os.path.join(data_path, file), variant=variant)
        )

    # import petri net model
    petri_nets = []
    for variant in variants:
        petri_nets.append(import_models(variant))

    # Fitness using Token-based replay
    token_based_replay(real_logs, petri_nets)

    # Precision using ETConformance method
    et_conformance(real_logs, petri_nets)

    # Earth Mover Distance
    earth_mover_distance(real_logs, petri_nets)
