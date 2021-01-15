import os
from conformance_checking import import_xes, import_petri_net, generate_playout

absPath = os.path.abspath(__file__)
fileDir = os.path.dirname(absPath)
code = os.path.dirname(fileDir)
data = os.path.join(code, "data")


print("Running the import_xes() Method to import an event log...")
log_data = import_xes(os.path.join(data, "log_test.xes"))
print(
    """ The output of the method is a List[List[str]],
     where the entry i,j is the j-th activity name of the i-th trace."""
)
print("Output: ", log_data)
print("Running the import_petri_net() Method to import a petri net...")
net, im, fm = import_petri_net(os.path.join(data, "petri_net_large.pnml"))
print("Aquired a petri net, an initial marking and a final marking.")
print("Running the generate_playout() Method to generate a playout of the petri net...")
playout = generate_playout(net, im, fm)
print(
    """The output of the method is a List[List[str]], where the entry i,j
    is the j-th activity name of the i-th trace."""
)
print("Output: ", playout)
