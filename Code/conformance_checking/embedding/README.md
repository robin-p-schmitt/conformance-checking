# Generating Embeddings for Activities and Traces

This is a tutorial for generating embeddings for activities and traces for a given log.

There are three possibilities for generating embeddings with a log:
 - generate activity and trace embeddings from a log at the same time
 - generate only activity embeddings from a log
 - generate only trace embeddings from a log
 
Make sure to have following imports before implementing any of the alternatives from above
```python
from conformance_checking.embedding.embedding_generator import Embedding_generator, Activity_Embedding_generator, Trace_Embedding_generator
from pm4py.objects.log.importer.xes import importer as xes_importer
```
 
 ## Generating Activity and Trace Embeddings from a Log at once

Following code creates an instance of an embedding generator which generates activity embedding and trace embedding
from a log at the same time
```python
# load a log with pm4py
log = xes_importer.apply("data/BPI_Challenge_2012.xes")

# only keep the first 2000 traces, so it is faster.
# If you want to test on the whole log, just remove the [:2000]
log = [[event["concept:name"] for event in trace] for trace in log][:2000]


# create instance of the embedding generator.
# paramaters are explained in the embedding_generator.py
emb_gen = Embedding_generator(
    log, trace2vec_windows_size=4, act2vec_windows_size=4, num_ns=4, activity_auto_train=True,
    trace_auto_train=True
)
```
 - Note that the parameter `activity_auto_train` is set to `true`, which triggers the generator to start training right
after an instance of the generator class is created. Analogous for `trace_auto-train`

 - If the `activity_auto_train` or `trace_auto_train` is set to `False`, make sure to train models before trying to
get embeddings, as following
```python
emb_gen.start_training()
```  

 - Once the model is trained, activity and trace embeddings can be retreived correspondingly as follows:
```python
# create example model and real log
model_log = log[:3]
real_log = log[3:8]

# get frequency tables for the model log and the real log
# and an embedding lookup table
model_freq, real_freq, embeddings = emb_gen.get_activity_embedding(
    model_log, real_log
    )

# get the trace embeddings of traces in the model log and real log
model_emb, real_emb = emb_gen.get_trace_embedding(model_log, real_log)
```
 - To get the frequency of activity with index 10 in the first trace from model_log:
```python
print(model_freq[0][10])
```
 - To get the list of dictionaries containing the counts of activities in traces from the real log:
 ```python
print(real_freq)
```

 - To get the embedding of the activity with index 0:
 ```python
print(embeddings[0])
```

 - To get the embedding of the first trace in the model log:
 ```python
print(model_emb[0])
```

## Generate only Activity/Trace Embedding from a log

Everything is analogous as the first example, but instead of creating an instance of `Embedding_generator`,
create either an instance of `Activity_Embedding_generator` or `Trace_Embedding_generator` as followings
```python
act_emb_gen = Activity_Embedding_generator(
        log, act2vec_windows_size=4, num_ns=4
    )
```
or
```python
trace_emb_gen = Trace_Embedding_generator(
        log, trace2vec_windows_size=4, auto_train=True
    )
```