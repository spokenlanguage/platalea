import torch
import numpy as np
import pickle
import logging
import platalea.basic as basic
import platalea.encoders as encoders
import platalea.dataset as dataset
import platalea.config
import json
import os


_device = platalea.config.device()


def prepare():
    logging.getLogger().setLevel('INFO')
    logging.info("Loading pytorch models")
    net_rand = basic.SpeechImage(basic.DEFAULT_CONFIG).to(_device)
    net_rand.eval()
    net_train = basic.SpeechImage(basic.DEFAULT_CONFIG)
    net_train.load_state_dict(torch.load("net.20.pt").state_dict())
    net_train.to(_device)
    net_train.eval()
    nets = [('trained', net_train), ('random', net_rand)]
    with torch.no_grad():
        save_data(nets, directory='.', batch_size=32)


def massage_transformer_data():
    logging.getLogger().setLevel('INFO')
    factors = pickle.load(open("data/in/trans/downsampling_factors.pkl", "rb"))
    layers = factors.keys()
    logging.info("Loading input")
    data = pickle.load(open("data/in/trans/global_input.pkl", "rb"))
    logging.info("Fixing IDs")
    data['audio_id'] = np.array([ i + '.wav' for i in data['audio_id']])
    logging.info("Adding IPA")
    alignment = load_alignment("data/out/trans/fa.json") 
    data['ipa'] = np.array([ align2ipa(alignment[i]) for i in data['audio_id'] ])
    logging.info("Saving input")
    pickle.dump(data, open("data/out/trans/global_input.pkl", "wb"), protocol=4)
    for mode in ['trained', 'random']:
        for layer in layers:
            logging.info("Loading data for {} {}".format(mode, layer))
            data = pickle.load(open("data/in/trans/global_{}.{}.pkl".format(mode, layer), "rb"))
            data = {layer: data}
            logging.info("Saving data for {} {}".format(mode, layer))
            pickle.dump(data, open("data/out/trans/global_{}_{}.pkl".format(mode, layer), "wb"), protocol=4)

def vgs_factors():
    return {'conv': {'pad': 0, 'ksize': 6, 'stride': 2 },
                   'rnn0': None,
                   'rnn1': None,
                   'rnn2': None,
                   'rnn3': None }

def massage_vgs_asr_data():
    logging.getLogger().setLevel('INFO')
    filter_global_data("data/in/vgs-asr/")
    factors = vgs_factors()
    layers = factors.keys()
    logging.info("Loading input")
    data = pickle.load(open("data/in/vgs-asr/global_input.pkl", "rb"))
    logging.info("Adding IPA")
    alignment = load_alignment("data/out/vgs-asr/fa.json") 
    data['ipa'] = np.array([ align2ipa(alignment[i]) for i in data['audio_id'] ])
    logging.info("Saving input")
    pickle.dump(data, open("data/out/vgs-asr/global_input.pkl", "wb"), protocol=4)
    for mode in ['trained', 'random']:
        logging.info("Loading data for {}".format(mode))
        data = pickle.load(open("data/in/vgs-asr/global_{}.pkl".format(mode), "rb"))
        for layer in layers:
            ldata = {layer: data[layer]}
            logging.info("Saving data for {} {}".format(mode, layer))
            pickle.dump(data, open("data/out/vgs-asr/global_{}_{}.pkl".format(mode, layer), "wb"), protocol=4)
        
def save_data(nets, directory, batch_size=32):
    save_global_data(nets, directory=directory, batch_size=batch_size) # FIXME adapt this per directory too
    save_local_data(directory=directory)
    
def save_global_data(nets,  directory='.', batch_size=32):
    """Generate data for training a phoneme decoding model."""
    logging.info("Loading alignments")
    data = load_alignment("{}/fa.json".format(directory))
    logging.info("Loading audio features")
    val = dataset.Flickr8KData(root='/roaming/gchrupal/datasets/flickr8k/', split='val')
    # 
    alignments = [ data[sent['audio_id']] for sent in val ]
    # Only consider cases where alignement does not fail
    alignments = [item for item in alignments if good_alignment(item) ]
    sentids = set(item['audio_id'] for item in alignments)
    audio = [ sent['audio'] for sent in val if sent['audio_id'] in sentids ]
    audio_np = [ a.numpy() for a in audio]

    ## Global data

    global_input = dict(audio_id = np.array([datum['audio_id'] for datum in alignments]),
                       ipa =      np.array([align2ipa(datum)  for datum in alignments]), 
                       text =     np.array([datum['transcript'] for datum in alignments]),
                       audio = np.array(audio_np))
    pickle.dump(global_input, open("global_input.pkl", "wb"), protocol=4)
    
    for mode, net in nets:
        global_act = collect_activations(net, audio, batch_size=batch_size)
        for layer in global_act:
            logging.info("Saving global data in {}/global_{}_{}.pkl".format(directory, mode, layer))
            pickle.dump({layer: global_act[layer]}, open("{}/global_{}_{}.pkl".format(directory, mode, layer), "wb"), protocol=4)

def filter_global_data(directory):
    """Remove sentences with OOV items from data."""
    logging.getLogger().setLevel('INFO')
    logging.info("Loading raw data")
    global_input =  pickle.load(open("{}/global_input_raw.pkl".format(directory), "rb"))
    logging.info("Loading alignments")
    adata = load_alignment("{}/fa.json".format(directory))

    logging.info("Filtering out failed alignments and OOV")
    alignments = [ adata.get(i, adata.get(i+'.wav')) for i in global_input['audio_id'] ]
    # Only consider cases where alignement does not fail
    # Only consider cases with no OOV items
    alignments = [item for item in alignments if good_alignment(item) ]
    sentids = set(item['audio_id'] for item in alignments)
    ## Global data

    include = np.array([ i in sentids for i in global_input['audio_id'] ])
    
    #filtered = { key: np.array(value)[include] for key, value in global_input.items() }
    # Hack to fix broken format
    filtered = {}
    for key, value in global_input.items():
        if key == "audio":
            value = [ v.numpy() for v in value ]
        filtered[key] = np.array(value)[include]
        
    logging.info("Saving filtered data")
    pickle.dump(filtered, open("{}/global_input.pkl".format(directory), "wb"), protocol=4)

    for name in ['trained', 'random']:
        global_act = pickle.load(open("{}/global_{}_raw.pkl".format(directory, name), "rb"))
        filtered = { key: np.array(value)[include] for key, value in global_act.items() }  
        logging.info("Saving filtered global data in global_{}.pkl".format(name))
        pickle.dump(filtered, open("{}/global_{}.pkl".format(directory, name), "wb"), protocol=4)

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
    
def save_local_data(directory):
    logging.getLogger().setLevel('INFO')
    logging.info("Loading alignments")
    global_input =  pickle.load(open("{}/global_input.pkl".format(directory), "rb"))
    adata = load_alignment("{}/fa.json".format(directory))
    alignments = [ adata.get(i, adata.get(i+'.wav')) for i in global_input['audio_id'] ]
    ## Local data
    local_data = {}
    logging.info("Computing local data for MFCC")
    y, X = phoneme_activations(global_input['audio'], alignments, index=lambda ms: ms//10, framewise=True)
    local_input = check_nan(features=X, labels=y)
    pickle.dump(local_input, open("{}/local_input.pkl".format(directory), "wb"), protocol=4)
    try:
        factors = pickle.load(open("{}/downsampling_factors.pkl".format(directory), "rb"))
    except FileNotFoundError:
        # Default VGS settings
        factors = vgs_factors()
    for mode in ['trained', 'random']:
        for layer in factors.keys():
            if layer == "conv1":
                pass # This data is too big
            else:
                global_act = pickle.load(open("{}/global_{}_{}.pkl".format(directory, mode, layer), "rb"))
                local_act = {}
                index = make_indexer(factors, layer)
                logging.info("Computing local data for {}, {}".format(mode, layer))
                y, X = phoneme_activations(global_act[layer], alignments, index=index, framewise=True)
                local_act[layer] = check_nan(features=X, labels=y)
                logging.info("Saving local data in local_{}_{}.pkl".format(mode, layer))
                pickle.dump(local_act, open("{}/local_{}_{}.pkl".format(directory, mode, layer), "wb"), protocol=4)

def load_alignment(path):
    data = {}
    for line in open(path):
        item = json.loads(line)
        item['audio_id'] = os.path.basename(item['audiopath'])
        data[item['audio_id']] = item  
    return data
            
def phoneme_activations(activations, alignments, index=lambda ms: ms//10, framewise=True):
    """Return array of phoneme labels and array of corresponding mean-pooled activation states."""
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
    from platalea.analysis.ipa import arpa2ipa
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
        assert index(start)<index(end)+1, "Something funny: {} {} {} {}".format(start, end, index(start), index(end))
        yield (phone, aggregate(rep[index(start):index(end)+1]))

def frames(utt, rep, index): 
     """Return pair sequence of (phoneme label, frame), given an 
        alignment object `utt`, a representation array `rep`, and 
       indexing function `index`. 
     """
     for phoneme in phones(utt): 
         phone, start, end = phoneme
         assert index(start)<index(end)+1, "Something funny: {} {} {} {}".format(start, end, index(start), index(end)) 
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
        act = net.SpeechEncoder.introspect(au.to(_device), l.to(_device))
        for k in act:
            if k not in out:
                out[k] = []
            out[k]  += [ item.detach().cpu().numpy() for item in act[k] ]
    return { k: np.array(v) for k,v in out.items() }
    

def spec(d):
    if type(d) != type(dict()):
        return type(d)
    else:
        return { key:spec(val) for key, val in d.items() }
