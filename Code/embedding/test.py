from embedding_generator import *

#log = xes_importer.apply('logs/BPI_Challenge_2012.xes')
emb_gen = Embedding_generator(3)
print(emb_gen.get_activity_embedding())
print(emb_gen.get_trace_embedding())