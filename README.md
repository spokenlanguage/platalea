# Platalea
Understanding visually grounded spoken language via multi-tasking

![install and run in conda env](https://github.com/egpbos/platalea/workflows/install%20and%20run%20in%20conda%20env/badge.svg?branch=master)
[![codecov](https://codecov.io/gh/spokenlanguage/platalea/branch/master/graph/badge.svg)](https://codecov.io/gh/spokenlanguage/platalea)



## Install

```sh
conda create -n platalea python==3.8 pytorch -c conda-forge -c pytorch
conda activate platalea
pip install torchvision
pip install .
```

## Run experiments

### Get data
Download the Flickr8k data.
For testing, we prepared the [Flickr1d subset](https://github.com/spokenlanguage/flickr1d) of 10 Flickr8k images and corresponding captions.

### Configuration
We use ConfigArgParse for setting necessary input variables.
This means you can use either a configuration file (config.ini or config.yml), environment variables or command line arguments to specify the necessary configuration parameters.
For instance, to use environment variables with the Flickr1d dataset, set the PLATALEA_DATA_ROOT variable to the directory in which the Flickr1d data was downloaded, e.g. if it was cloned at `/user/data/flickr1d` do:

```sh
export FLICKR8K_ROOT=/user/data/flickr1d
export FLICKR8K_METADATA_JSON=dataset.json
export PLATALEA_AUDIO_FEATURES_FN=mfcc_features.pt
```

You could also specify these options on the command line directly when running (the respective options would be `--flickr8k_root=...`, `--flickr8k_meta=...` and `--audio_features_fn=...`) or via a configuration file (using the names of the command line options without leading `--`).

### Experiment dependencies
Different experiments may have different additional dependencies.
The `basic-stack` experiment needs the following:

```sh
pip install sklearn python-Levenshtein
```

### Run
Then finally, run the experiment:

```sh
python experiments/basic-stack/run.py
```
