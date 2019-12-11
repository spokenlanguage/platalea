# Data format

For analyzing the VGS model we are using the data files described below. For each new model we'll need similar files, with the structure adapted to the architecture of the model if needed. The data is split into several files due to it's large size.

## Global 
```
global_input.pkl
{'audio_id': <class 'numpy.ndarray'>, 
 'audio': <class 'numpy.ndarray'>, 
 'text': <class 'numpy.ndarray'>}

global_trained_LAYER.pkl
{'LAYER': <class 'numpy.ndarray'> }

global_random_LAYER.pkl
{'LAYER': <class 'numpy.ndarray'> }

downsampling_factors.json

fa.json

```
