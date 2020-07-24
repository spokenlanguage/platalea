import json
import logging
import os
import numpy as np
from pathlib import Path
import pickle
import platalea.dataset as dataset
from prepare_flickr8k import align2ipa, check_nan, good_alignment, \
        load_alignment, make_indexer, phoneme_activations


def save_data(indir, outdir, mode):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    save_global_data(indir, outdir=outdir, mode=mode,
                     alignment_fpath='data/datasets/flickr8k/fa.json')
    save_local_data(outdir=outdir, mode=mode,
                    alignment_fpath='data/datasets/flickr8k/fa.json')


def save_data_trigrams(indir, outdir, mode):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    save_global_data_trigrams(indir, outdir=outdir, mode=mode,
                              alignment_fpath='data/flickr8k_trigrams_fa.json')
    save_local_data(outdir=outdir, mode=mode,
                    alignment_fpath="data/flickr8k_trigrams_fa.json")


def save_global_data_trigrams(indir, outdir, mode, alignment_fpath):
    """Generate data from trigrams for training a phoneme decoding model."""
    from platalea.vq_encode import config, audio_features
    logging.info("Loading alignments")
    data = load_alignment(alignment_fpath)
    # Only consider cases where alignement does not fail
    alignments = [item for item in data.values() if good_alignment(item)]
    paths = [item['audiopath'] for item in alignments]
    audio = audio_features(paths, config)
    audio_np = [a.numpy() for a in audio]

    ## Global data
    audio_id = np.array([datum['audio_id'] for datum in alignments])
    global_input = dict(
        audio_id=audio_id,
        ipa=np.array([align2ipa(datum) for datum in alignments]),
        text=np.array([datum['transcript'] for datum in alignments]),
        audio=np.array(audio_np))
    global_input_path = Path(outdir) / 'global_input.pkl'
    pickle.dump(global_input, open(global_input_path, "wb"), protocol=4)

    # Global activations
    encodings = []
    for sid in audio_id:
        fpath = os.path.join(indir, os.path.splitext(sid)[0] + '.txt')
        encodings.append(np.loadtxt(fpath))
    path = "{}/global_{}_codebook.pkl".format(outdir, mode)
    logging.info("Saving global data in {}".format(path))
    pickle.dump({'codebook': encodings}, open(path, "wb"), protocol=4)


def save_global_data(indir, outdir, mode, alignment_fpath):
    """Generate data for training a phoneme decoding model."""
    logging.info("Loading alignments")
    data = load_alignment(alignment_fpath)
    logging.info("Loading audio features")
    val = dataset.Flickr8KData(root='data/datasets/flickr8k/', split='val',
                               feature_fname='mfcc_delta_features.pt')
    # Vocabulary should be initialized even if we are not going to use text
    # data
    if dataset.Flickr8KData.le is None:
        dataset.Flickr8KData.init_vocabulary(val)
    alignments = [data[sent['audio_id']] for sent in val]
    # Only consider cases where alignement does not fail
    alignments = [item for item in alignments if good_alignment(item)]
    sentids = set(item['audio_id'] for item in alignments)
    audio = [sent['audio'] for sent in val if sent['audio_id'] in sentids]
    audio_np = [a.numpy() for a in audio]

    # Global input
    audio_id = np.array([datum['audio_id'] for datum in alignments])
    global_input = dict(
        audio_id=audio_id,
        ipa=np.array([align2ipa(datum) for datum in alignments]),
        text=np.array([datum['transcript'] for datum in alignments]),
        audio=np.array(audio_np))
    global_input_path = Path(outdir) / 'global_input.pkl'
    pickle.dump(global_input, open(global_input_path, "wb"), protocol=4)

    # Global activations
    encodings = []
    for sid in audio_id:
        fpath = os.path.join(indir, os.path.splitext(sid)[0] + '.txt')
        encodings.append(np.loadtxt(fpath))
    path = "{}/global_{}_codebook.pkl".format(outdir, mode)
    logging.info("Saving global data in {}".format(path))
    pickle.dump({'codebook': encodings}, open(path, "wb"), protocol=4)


def save_local_data(outdir, mode, alignment_fpath):
    logging.getLogger().setLevel('INFO')
    logging.info("Loading alignments")
    global_input = pickle.load(open("{}/global_input.pkl".format(outdir),
                                    "rb"))
    adata = load_alignment(alignment_fpath)
    alignments = [adata[i] for i in global_input['audio_id']]
    # Local data
    logging.info("Computing local data for MFCC")
    y, X = phoneme_activations(global_input['audio'], alignments,
                               index=lambda ms: ms//10, framewise=True)
    local_input = check_nan(features=X, labels=y)
    pickle.dump(local_input, open("{}/local_input.pkl".format(outdir), "wb"), protocol=4)
    factors = json.load(open("{}/downsampling_factors.json".format(outdir), "rb"))
    for layer in factors.keys():
        if layer[:4] == "conv":
            pass  # This data is too big
        else:
            global_act = pickle.load(open("{}/global_{}_{}.pkl".format(outdir, mode, layer), "rb"))
            local_act = {}
            index = make_indexer(factors, layer)
            logging.info("Computing local data for {}, {}".format(mode, layer))
            y, X = phoneme_activations(global_act[layer], alignments,
                                       index=index, framewise=True)
            local_act[layer] = check_nan(features=X, labels=y)
            logging.info("Saving local data in local_{}_{}.pkl".format(mode, layer))
            pickle.dump(local_act, open("{}/local_{}_{}.pkl".format(outdir, mode, layer), "wb"), protocol=4)
