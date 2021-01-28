import os
from conformance_checking import import_petri_net, generate_playout

absPath = os.path.abspath(__file__)
fileDir = os.path.dirname(absPath)
code = os.path.dirname(fileDir)
data = os.path.join(code, "data")


def test_import_petri_net():
    net, im, fm = import_petri_net(os.path.join(data, "petri_net_small.pnml"))

    assert str(im) == "['source:1']"
    assert str(fm) == "['sink:1']"


def test_generate_playout():
    net, im, fm = import_petri_net(os.path.join(data, "petri_net_small.pnml"))
    playout = generate_playout(net, im, fm, "concept:name")
    assert len(playout) == 1000
    max = 0
    for i in playout:
        if len(i) > max:
            max = len(i)
    assert max == 1
