import torch
import numpy as np
import pickle
import logging
import platalea.basic as basic
import platalea.encoders as encoders
import platalea.dataset as dataset
import json
import os

def prepare():
    logging.getLogger().setLevel('INFO')
    logging.info("Loading pytorch models")
    net_rand = basic.SpeechImage(basic.DEFAULT_CONFIG).cuda()
    net_rand.eval()
    net_train = basic.SpeechImage(basic.DEFAULT_CONFIG)
    net_train.load_state_dict(torch.load("experiments/basic-stack/net.20.pt").state_dict())
    net_train.cuda()
    net_train.eval()
    nets = [('trained', net_train), ('random', net_rand)]
    with torch.no_grad():
        save_data(nets, batch_size=32, framewise=True)

def save_data(nets,  batch_size, framewise=True):
    """Generate data for training a phoneme decoding model."""

    alignment_path="/roaming/gchrupal/datasets/flickr8k/dataset.val.fa.json"    
    logging.info("Loading alignments")
    data = {}
    for line in open(alignment_path):
        item = json.loads(line)
        item['audio_id'] = os.path.basename(item['audiopath'])
        data[item['audio_id']] = item
    logging.info("Loading audio features")
    val = dataset.Flickr8KData(root='/roaming/gchrupal/datasets/flickr8k/', split='val')
    # 
    alignments_all = [ data[sent['audio_id']] for sent in val ]
    # Only consider cases where alignement does not fail
    alignments = [ item for item in alignments_all if np.all([word.get('start', False) for word in item['words']]) ]
    sentids = set(item['audio_id'] for item in alignments)
    audio = [ sent['audio'] for sent in val if sent['audio_id'] in sentids ]
    audio_np = [ a.numpy() for a in audio]

    ## Global data

    global_input = dict(audio_id = np.array([datum['audio_id'] for datum in alignments]),
                       ipa =      np.array([align2ipa(datum)  for datum in alignments]), 
                       text =     np.array([datum['transcript'] for datum in alignments]),
                       audio = np.array(audio_np))
    pickle.dump(global_input, open("global_input.pkl", "wb"), protocol=4)
    
    for name, net in nets:
        global_act = collect_activations(net, audio, batch_size=batch_size)
        logging.info("Saving global data in global_{}.pkl".format(name))
        pickle.dump(global_act, open("global_{}.pkl".format(name), "wb"), protocol=4)
        
    ## Local data
    local_data = {}
    logging.info("Computing local data for MFCC")
    y, X = phoneme_activations(global_input['audio'], alignments, index=lambda ms: ms//10, framewise=framewise)
    local_input = check_nan(features=X, labels=y)
    pickle.dump(local_input, open("local_input.pkl", "wb"), protocol=4)
    for name, net in nets:
        local_act = {}
        index = lambda ms: encoders.inout(net.SpeechEncoder.Conv, torch.tensor(ms)//10).numpy()
        global_act = pickle.load(open("global_{}.pkl".format(name), "rb"))
        for key in ['conv', 'rnn0', 'rnn1', 'rnn2', 'rnn3']:
                logging.info("Computing local data for {}, {}".format(name, key))
                y, X = phoneme_activations(global_act[key], alignments, index=index)
                local_act[key] = check_nan(features=X, labels=y)
        logging.info("Saving local data in local_{}.pkl".format(name))
        pickle.dump(local_act, open("local_{}.pkl".format(name), "wb"), protocol=4)

        
            
def phoneme_activations(activations, alignments, index=lambda ms: ms//10, framewise=True):
    """Return array of phoneme labels and array of corresponding mean-pooled activation states."""
    labels = []
    states = []
    for activation, alignment in zip(activations, alignments):
        # extract phoneme labels and activations for current utterance
        if framewise:
            y, X = zip(*list(frames(alignment, activation, index=index)))
        else:
            y, X = zip(*list(slices(alignment, activation, index=index)))
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
             yield (phone, rep[j])
             
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
        act = net.SpeechEncoder.introspect(au.cuda(), l.cuda())
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
