from conformance_checking import import_xes, import_petri_net, generate_playout
from conformance_checking.algorithms import Act2VecWmdConformance


def main():
    real_log = import_xes("data/benchmark/GTlog-11.xes", "concept:name")
    model_log = generate_playout(
        *import_petri_net("data/benchmark/alpha-11.pnml"), "concept:name"
    )
    result = Act2VecWmdConformance().execute(model_log, real_log)
    print(result.calc_precision())
    print(result.calc_fitness())


if __name__ == "__main__":
    main()
