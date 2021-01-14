from embedding_generator import *

# just gave a random log, since the class initializer already has log for testing purpose
emb_gen = Embedding_generator(3)
print(emb_gen.get_activity_embedding())
print(emb_gen.get_trace_embedding())