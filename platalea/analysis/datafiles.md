# Data format

For analyzing the VGS model we are using the data files described below. For each new model we'll need similar files, with the structure adapted to the architecture of the model if needed.

## Global 
`global_data.pkl` data for global methods. The data is the dictionary with the following structure:

```
{'audio_id': numpy.ndarray,
 'ipa': numpy.ndarray,
 'text': numpy.ndarray,
 'audio': numpy.ndarray,
 'trained': {'conv': numpy.ndarray,
  'rnn0': numpy.ndarray,
  'rnn1': numpy.ndarray,
  'rnn2': numpy.ndarray,
  'rnn3': numpy.ndarray,
  'att': numpy.ndarray},
 'random': {'conv': numpy.ndarray,
  'rnn0': numpy.ndarray,
  'rnn1': numpy.ndarray,
  'rnn2': numpy.ndarray,
  'rnn3': numpy.ndarray,
  'att': numpy.ndarray}}
  ```
  All the data in the arrays are in the same order.
  
 ## Local 
 `local_data.pkl` data for local methods. The data is the dictionary with the following structure:
 ``` 
{'mfcc': {'features': numpy.ndarray, 'labels': numpy.ndarray},
 'random': {'conv': {'features': numpy.ndarray, 'labels': numpy.ndarray},
  'rnn0': {'features': numpy.ndarray, 'labels': numpy.ndarray},
  'rnn1': {'features': numpy.ndarray, 'labels': numpy.ndarray},
  'rnn2': {'features': numpy.ndarray, 'labels': numpy.ndarray},
  'rnn3': {'features': numpy.ndarray, 'labels': numpy.ndarray}},
 'trained': {'conv': {'features': numpy.ndarray, 'labels': numpy.ndarray},
  'rnn0': {'features': numpy.ndarray, 'labels': numpy.ndarray},
  'rnn1': {'features': numpy.ndarray, 'labels': numpy.ndarray},
  'rnn2': {'features': numpy.ndarray, 'labels': numpy.ndarray},
  'rnn3': {'features': numpy.ndarray, 'labels': numpy.ndarray}}}
  ```
In these dictionaries 'features' correspond to inputs or activations for a single frame or timestep. The 'labels' array contains the corresponding phoneme labels, based on forced alignment between input audio and its transcription.
  
