# Data format

- `global_data.pkl` data for global methods. The data is the dictionary with the following structure:
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
  
  - `local_data.pkl` data for local methods. The data is the dictionary with the following structure:
  
