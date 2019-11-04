# Data format

For analyzing the VGS model we are using the data files described below. For each new model we'll need similar files, with the structure adapted to the architecture of the model if needed. The data is split into several files due to it's large size.

## Global 
```
global_input.pkl
{'audio_id': <class 'numpy.ndarray'>, 
 'ipa': <class 'numpy.ndarray'>, 
 'text': <class 'numpy.ndarray'>, 
 'audio': <class 'numpy.ndarray'>}

global_trained.pkl
{'conv': <class 'numpy.ndarray'>, 
 'rnn0': <class 'numpy.ndarray'>, 
 'rnn1': <class 'numpy.ndarray'>, 
 'rnn2': <class 'numpy.ndarray'>, 
 'rnn3': <class 'numpy.ndarray'>, 
 'att': <class 'numpy.ndarray'>}

global_random.pkl
{'conv': <class 'numpy.ndarray'>, 
 'rnn0': <class 'numpy.ndarray'>, 
 'rnn1': <class 'numpy.ndarray'>, 
 'rnn2': <class 'numpy.ndarray'>, 
 'rnn3': <class 'numpy.ndarray'>, 
 'att': <class 'numpy.ndarray'>}
```

## Local

```
local_input.pkl
{'features': <class 'numpy.ndarray'>, 'labels': <class 'numpy.ndarray'>}

local_trained.pkl
{'conv': {'features': <class 'numpy.ndarray'>, 'labels': <class 'numpy.ndarray'>}, 
 'rnn0': {'features': <class 'numpy.ndarray'>, 'labels': <class 'numpy.ndarray'>}, 
 'rnn1': {'features': <class 'numpy.ndarray'>, 'labels': <class 'numpy.ndarray'>}, 
 'rnn2': {'features': <class 'numpy.ndarray'>, 'labels': <class 'numpy.ndarray'>}, 
 'rnn3': {'features': <class 'numpy.ndarray'>, 'labels': <class 'numpy.ndarray'>}}

local_random.pkl
{'conv': {'features': <class 'numpy.ndarray'>, 'labels': <class 'numpy.ndarray'>}, 
 'rnn0': {'features': <class 'numpy.ndarray'>, 'labels': <class 'numpy.ndarray'>}, 
 'rnn1': {'features': <class 'numpy.ndarray'>, 'labels': <class 'numpy.ndarray'>}, 
 'rnn2': {'features': <class 'numpy.ndarray'>, 'labels': <class 'numpy.ndarray'>}, 
 'rnn3': {'features': <class 'numpy.ndarray'>, 'labels': <class 'numpy.ndarray'>}}

```


In these dictionaries 'features' correspond to inputs or activations for a single frame or timestep. The 'labels' array contains the corresponding phoneme labels, based on forced alignment between input audio and its transcription. Note that in this dataset some utterances have been filtered out due to the force alignment failing.
  
