import os
from conformance_checking import import_xes

absPath = os.path.abspath(__file__)
fileDir = os.path.dirname(absPath)
code = os.path.dirname(fileDir)
data = os.path.join(code, "data")


def test_import_xes():
    assert import_xes(os.path.join(data, "log_test.xes")) == [
        ["register request", "examine casually", "check ticket"],
        ["register request", "decide"],
    ], str(data)
