# Representations of phonemes

We consider two analytical approaches:

- **diagnostic classification**, where neural activations are used as input features to a classifier which predicts which phonemes they 
  correspond to;
- **representational similarity analysis**, where similarities are measured in phoneme or phoneme-string space and compared to 
  similarities in neural activation space.
  
We also consider two ways of viewing activation patterns in hidden layers as representations:

- **local activations**: we use single frames/time-steps, or short segments as neural representations
- **pooled activations**: pool activations over the whole utterance to use as a global neural representation.

