# Platalea
Understanding visually grounded spoken language via multi-tasking

![install and run in conda env](https://github.com/egpbos/platalea/workflows/install%20and%20run%20in%20conda%20env/badge.svg?branch=master)

## Install

```sh
conda create -n platalea python==3.8 pytorch -c conda-forge -c pytorch
conda activate platalea
pip install torchvision
pip install .
```

## Run experiments

Download the Flickr8k data.

Copy `config_template.py` to `config.py` and modify the values to match the Flickr8k download path.

Different experiments may have different additional dependencies.
The `basic-stack` experiment needs the following:

```sh
pip install sklearn python-Levenshtein
```

Then run as

```sh
python experiments/basic-stack/run.py
```
