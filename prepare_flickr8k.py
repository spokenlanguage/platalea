import pickle
import logging
import platalea.dataset as dataset
import json
import os
from pathlib import Path
import torch
import numpy as np


def save_data(nets, directory, batch_size=32):
    Path(directory).mkdir(parents=True, exist_ok=True)
    save_global_data(nets, directory=directory,
                     alignment_fpath='data/datasets/flickr8k/fa.json',
                     batch_size=batch_size)  # FIXME adapt this per directory too
    save_local_data(directory=directory,
                    alignment_fpath='data/datasets/flickr8k/fa.json')


def save_data_trigrams(nets, directory, batch_size=32):
    Path(directory).mkdir(parents=True, exist_ok=True)
    json.dump(make_factors(nets[0][1]), open(Path(directory) / "downsampling_factors.json", "w"))
    save_global_data_trigrams(nets, directory=directory,
                              alignment_fpath='data/flickr8k_trigrams_fa.json',
                              batch_size=batch_size)  # FIXME adapt this per directory too
    save_local_data(directory=directory,
                    alignment_fpath="data/flickr8k_trigrams_fa.json")


def save_global_data_trigrams(nets, directory, alignment_fpath, batch_size=32):
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

    global_input = dict(
        audio_id=np.array([datum['audio_id'] for datum in alignments]),
        ipa=np.array([align2ipa(datum) for datum in alignments]),
        text=np.array([datum['transcript'] for datum in alignments]),
        audio=np.array(audio_np))
    global_input_path = Path(directory) / 'global_input.pkl'
    pickle.dump(global_input, open(global_input_path, "wb"), protocol=4)

    for mode, net in nets:
        global_act = collect_activations(net, audio, batch_size=batch_size)
        for layer in global_act:
            path = "{}/global_{}_{}.pkl".format(directory, mode, layer)
            logging.info("Saving global data in {}".format(path))
            pickle.dump({layer: global_act[layer]}, open(path, "wb"),
                        protocol=4)


def save_global_data(nets, directory, alignment_fpath, batch_size=32):
    """Generate data for training a phoneme decoding model."""
    logging.info("Loading alignments")
    data = load_alignment(alignment_fpath)
    logging.info("Loading audio features")
    val = dataset.Flickr8KData(root='data/datasets/flickr8k/', split='val',
                               feature_fname='mfcc_features.pt')
    # Vocabulary should be initialized even if we are not going to use text
    # data
    if dataset.Flickr8KData.le is None:
        dataset.Flickr8KData.init_vocabulary(val)
    #
    alignments = [data[sent['audio_id']] for sent in val]
    # Only consider cases where alignement does not fail
    alignments = [item for item in alignments if good_alignment(item)]
    sentids = set(item['audio_id'] for item in alignments)
    audio = [sent['audio'] for sent in val if sent['audio_id'] in sentids]
    audio_np = [a.numpy() for a in audio]

    ## Global data

    global_input = dict(
        audio_id=np.array([datum['audio_id'] for datum in alignments]),
        ipa=np.array([align2ipa(datum) for datum in alignments]),
        text=np.array([datum['transcript'] for datum in alignments]),
        audio=np.array(audio_np))
    global_input_path = Path(directory) / 'global_input.pkl'
    pickle.dump(global_input, open(global_input_path, "wb"), protocol=4)

    for mode, net in nets:
        global_act = collect_activations(net, audio, batch_size=batch_size)
        for layer in global_act:
            path = "{}/global_{}_{}.pkl".format(directory, mode, layer)
            logging.info("Saving global data in {}".format(path))
            pickle.dump({layer: global_act[layer]}, open(path, "wb"),
                        protocol=4)


def good_alignment(item):
    for word in item['words']:
        if word['case'] != 'success' or word['alignedWord'] == '<unk>':
            return False
    return True


def make_indexer(factors, layer):
    def inout(pad, ksize, stride, L):
        return ((L + 2*pad - 1*(ksize-1) - 1) // stride + 1)

    def index(ms):
        t = ms//10
        for l in factors:
            if factors[l] is None:
                pass
            else:
                pad = factors[l]['pad']
                ksize = factors[l]['ksize']
                stride = factors[l]['stride']
                t = inout(pad, ksize, stride, t)
            if l == layer:
                break
        return t

    return index


def save_local_data(directory, alignment_fpath):
    logging.getLogger().setLevel('INFO')
    logging.info("Loading alignments")
    global_input = pickle.load(open("{}/global_input.pkl".format(directory),
                                    "rb"))
    adata = load_alignment(alignment_fpath)
    alignments = [adata.get(i, adata.get(i+'.wav')) for i in global_input['audio_id']]
    ## Local data
    logging.info("Computing local data for MFCC")
    y, X = phoneme_activations(global_input['audio'], alignments,
                               index=lambda ms: ms//10, framewise=True)
    local_input = check_nan(features=X, labels=y)
    pickle.dump(local_input, open("{}/local_input.pkl".format(directory), "wb"), protocol=4)
    try:
        factors = json.load(open("{}/downsampling_factors.json".format(directory), "rb"))
    except FileNotFoundError:
        # Default VGS settings
        factors = default_factors()
    for mode in ['trained', 'random']:
        for layer in factors.keys():
            if layer == "conv1" or layer == "conv2":
                pass  # This data is too big
            else:
                global_act = pickle.load(open("{}/global_{}_{}.pkl".format(directory, mode, layer), "rb"))
                local_act = {}
                index = make_indexer(factors, layer)
                logging.info("Computing local data for {}, {}".format(mode, layer))
                y, X = phoneme_activations(global_act[layer], alignments,
                                           index=index, framewise=True)
                local_act[layer] = check_nan(features=X, labels=y)
                logging.info("Saving local data in local_{}_{}.pkl".format(mode, layer))
                pickle.dump(local_act, open("{}/local_{}_{}.pkl".format(directory, mode, layer), "wb"), protocol=4)


def default_factors():
    return dict(conv=dict(pad=0, ksize=6, stride=2),
                rnn_bottom0=None,
                codebook=None,
                rnn_top0=None,
                rnn_top1=None,
                rnn_top2=None)


def make_factors(net):
    conv = dict(pad=0,
                ksize=net.SpeechEncoder.Bottom.Conv.kernel_size[0],
                stride=net.SpeechEncoder.Bottom.Conv.stride[0])
    D = dict(conv=conv)
    for k in range(net.SpeechEncoder.Bottom.RNN.num_layers):
        D['rnn_bottom{}'.format(k)] = None
    D['codebook'] = None
    for k in range(net.SpeechEncoder.Top.RNN.num_layers):
        D['rnn_top{}'.format(k)] = None
    return D


def load_alignment(path):
    data = {}
    for line in open(path):
        item = json.loads(line)
        item['audio_id'] = os.path.basename(item['audiopath'])
        data[item['audio_id']] = item
    return data


def phoneme_activations(activations, alignments, index=lambda ms: ms//10,
                        framewise=True):
    """Return array of phoneme labels and array of corresponding mean-pooled
    activation states."""
    labels = []
    states = []
    for activation, alignment in zip(activations, alignments):
        # extract phoneme labels and activations for current utterance
        if framewise:
            fr = list(frames(alignment, activation, index=index))
        else:
            fr = list(slices(alignment, activation, index=index))
        if len(fr) > 0:
            y, X = zip(*fr)
            y = np.array(y)
            X = np.stack(X)
            labels.append(y)
            states.append(X)
    return np.concatenate(labels), np.concatenate(states)


def align2ipa(datum):
    """Extract IPA transcription from alignment information for a sentence."""
    from platalea.ipa import arpa2ipa
    result = []
    for word in datum['words']:
        for phoneme in word['phones']:
            result.append(arpa2ipa(phoneme['phone'].split('_')[0], '_'))
    return ''.join(result)


def slices(utt, rep, index, aggregate=lambda x: x.mean(axis=0)):
    """Return sequence of slices associated with phoneme labels, given an
       alignment object `utt`, a representation array `rep`, and
       indexing function `index`, and an aggregating function\
       `aggregate`.
    """
    for phoneme in phones(utt):
        phone, start, end = phoneme
        assert index(start) < index(end)+1, "Something funny: {} {} {} {}".format(start, end, index(start), index(end))
        yield (phone, aggregate(rep[index(start):index(end)+1]))


def frames(utt, rep, index):
    """Return pair sequence of (phoneme label, frame), given an
       alignment object `utt`, a representation array `rep`, and
      indexing function `index`.
    """
    for phoneme in phones(utt):
        phone, start, end = phoneme
        assert index(start) < index(end)+1, "Something funny: {} {} {} {}".format(start, end, index(start), index(end))
        for j in range(index(start), index(end)+1):
            if j < rep.shape[0]:
                yield (phone, rep[j])
            else:
                logging.warning("Index out of bounds: {} {}".format(j, rep.shape))


def phones(utt):
    """Return sequence of phoneme labels associated with start and end
     time corresponding to the alignment JSON object `utt`.

    """
    for word in utt['words']:
        pos = word['start']
        for phone in word['phones']:
            start = pos
            end = pos + phone['duration']
            pos = end
            label = phone['phone'].split('_')[0]
            if label != 'oov':
                yield (label, int(start*1000), int(end*1000))


def check_nan(labels, features):
    # Get rid of NaNs
    ix = np.isnan(features.sum(axis=1))
    logging.info("Found {} NaNs".format(sum(ix)))
    X = features[~ix]
    y = labels[~ix]
    return dict(features=X, labels=y)


def collect_activations(net, audio, batch_size=32):
    data = torch.utils.data.DataLoader(dataset=audio,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=0,
                                       collate_fn=dataset.batch_audio)
    out = {}
    for au, l in data:
        logging.info("Introspecting a batch of size {}x{}x{}".format(au.shape[0], au.shape[1], au.shape[2]))
        act = net.SpeechEncoder.introspect(au.cuda(), l.cuda())
        #print({k:[vi.shape for vi in v] for k,v in act.items()})
        for k in act:
            if k not in out:
                out[k] = []
            out[k] += [item.detach().cpu().numpy() for item in act[k]]

    return {k: np.array(v) for k, v in out.items()}


def spec(d):
    if type(d) != type(dict()):
        return type(d)
    else:
        return {key: spec(val) for key, val in d.items()}
