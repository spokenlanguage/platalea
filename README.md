# Platalea
Understanding visually grounded spoken language via multi-tasking

[![DOI](https://zenodo.org/badge/239750248.svg)](https://zenodo.org/badge/latestdoi/239750248)
[![install and run tests](https://github.com/egpbos/platalea/workflows/install%20and%20run%20tests/badge.svg?branch=master)](https://github.com/spokenlanguage/platalea/actions/workflows/pythonapp.yml)
[![codecov](https://codecov.io/gh/spokenlanguage/platalea/branch/master/graph/badge.svg)](https://codecov.io/gh/spokenlanguage/platalea)

## Installation

Clone this repo and cd into it:

```sh
git clone https://github.com/spokenlanguage/platalea.git
cd platalea
```

To install in a conda environment, assuming conda has already been installed, run the following to download and install dependencies:

```sh
conda create -n platalea python==3.8 pytorch -c conda-forge -c pytorch
conda activate platalea
pip install torchvision
```

Then install platalea with:

```sh
pip install .
```

### Experiment dependencies
Different experiments may have different additional dependencies.
The `basic` experiment needs the following:

```sh
pip install sklearn python-Levenshtein
```

## Dataset

The repository has been developed to work with Flickr8K dataset. The code can
be made to work with other datasets but this will require some adaptations.

To use Flickr8K, you need to download:
* [Flickr8K](http://hockenmaier.cs.illinois.edu/Framing_Image_Description/KCCA.html) [1].
  Note that downloading from the official website seems broken at the moment.
  Alternatively, the dataset can be obtained from
  [here](https://github.com/jbrownlee/Datasets/blob/master/Flickr8k_Dataset.names).
* The [Flickr Audio Caption Corpus](https://groups.csail.mit.edu/sls/downloads/flickraudio/) [2].
* Some additional [metadata files](https://surfdrive.surf.nl/files/index.php/s/EF1bA9YYfhiBxoN).

Create a folder to store the dataset (we will assume here that the folder is
`~/corpora/flickr8k`)  and move all the files you downloaded there, then
extract the content of the archives.  You can now setup the environment and
start preprocessing the data.

### Configuration

We use ConfigArgParse for setting necessary input variables, including the
location of the dataset.  This means you can use either a configuration file
(config.ini or config.yml), environment variables or command line arguments to
specify the necessary configuration parameters.

To specify the location of the dataset, one option is to create a configuration
file under your home directory (`~/.config/platalea/config.yml`)., with
follwing content:

```
flickr8k_root   /home/<user>/corpora/flickr8k
```

The same result can be achieved with an environment variable:

```sh
export FLICKR8K_ROOT=/home/<user>/corpora/flickr8k
```

You could also specify this option directly on the command line when running
an experiment (the respective options would be `--flickr8k_root=...`).

### Preprocessing

Run the preprocessing script to extract input features:

```bash
python platalea/utils/preprocessing.py flickr8k
```

## Training

You can now train a model using one of the examples provided under
`platalea/experiments`, e.g.:

```sh
cd platalea/experiments/flickr8k
mkdir -p runs/test
cd runs/test
python -m platalea.experiments.flickr8k.basic
```

After the model is trained, results are available in `results.json`.

### Weights and Biases (wandb)

Some experiments support the use of wandb for cloud logging of results.
In the examples we provide under `platalea/experiments`, this option is disabled by default.
To enable it, the call to `experiment()` should be changed from `experiment(..., wandb_mode='disabled')` to experiment(..., wandb_mode='online'). To default back to wandb normal behavior (where the mode can be set through command line or environment variable), use `wandb_mode=None` (or ignore the parameter).

## Contributing

If you want to contribute to the development of platalea, have a look at the [contribution guidelines](CONTRIBUTING.md).

## Changelog

We keep track of what is added, changed and removed in releases in the [changelog](CHANGELOG.md).

## References

[1] Hodosh, M., Young, P., & Hockenmaier, J. (2013). Framing Image Description
as a Ranking Task: Data, Models and Evaluation Metrics. Journal of Artificial
Intelligence Research, 47, 853–899. https://doi.org/10.1613/jair.3994.

[2] Harwath, D., & Glass, J. (2015). Deep multimodal semantic embeddings for
speech and images. 2015 IEEE Workshop on Automatic Speech Recognition and
Understanding (ASRU), 237–244. https://doi.org/10.1109/ASRU.2015.7404800.