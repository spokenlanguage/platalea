import sys
import torch
torch.manual_seed(123)
import platalea.basic as basic
import platalea.encoders as encoders
import platalea.dataset as dataset
import os.path
import logging
import json
import numpy as np
from plotnine import *
import pandas as pd 
import ursa.similarity as S
import ursa.util as U

def phoneme_decoding():
    logging.getLogger().setLevel('INFO')
    logging.info("Loading pytorch models")
    net_rand = basic.SpeechImage(basic.DEFAULT_CONFIG).cuda()
    net_train = basic.SpeechImage(basic.DEFAULT_CONFIG)
    net_train.load_state_dict(torch.load("experiments/basic-stack/net.20.pt").state_dict())
    net_train.cuda()
    #nets = [('random', net_rand), ('trained', net_train)]
    nets = [('trained', net_train), ('random', net_rand)]
 
    result = []
    # Recurrent
    for rep, net in nets:
        with torch.no_grad():
            net.eval()
            data = phoneme_data([(rep, net)], batch_size=32) # FIXME this hack is to prevent RAM error
            np.save('phoneme_data_{}.npy'.format(rep), data)
            logging.info("Fitting Logistic Regression for mfcc")
            acc, w = logreg_acc(data['mfcc']['features'], data['mfcc']['labels'])
            logging.info("Result for {}, {} = {}".format(rep, 'mfcc', acc))
            np.save('logreg_w_{}_{}.npy'.format(rep, 'mfcc'), w)
            result.append(dict(model=rep, layer='mfcc', layer_id=0, acc=acc))
            for j, kind in enumerate(data[rep], start=1):
                logging.info("Fitting Logistic Regression for {}, {}".format(rep, kind))
                acc, w = logreg_acc(data[rep][kind]['features'], data[rep][kind]['labels'])
                logging.info("Result for {}, {} = {}".format(rep, kind, acc))
                np.save('logreg_w_{}_{}.npy'.format(rep, kind), w)
                result.append(dict(model=rep, layer=kind, layer_id=j, acc=acc))
    json.dump(result, open("experiments/basic-stack/phoneme_decoding.json", "w"), indent=True)
    data = pd.read_json("experiments/basic-stack/phoneme_decoding.json", orient='records') 
    g = ggplot(data, aes(x='layer_id', y='acc', color='model')) + geom_point(size=2) + geom_line(size=2) + ylim(0,max(data['acc'])) 
    ggsave(g, 'experiments/basic-stack/phoneme_decoding.png')
     
def phoneme_data(nets,  batch_size):
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
    phon  = [ {'audio_id': datum['audio_id'],
               'ipa': align2ipa(datum),
               'transcript': datum['transcript'],
               'audio': au } for datum, au in zip(alignments, audio_np) ]
    logging.info("Saving IPA transcriptions")
    np.save("transcription.val.npy", phon)
    result = {}
    logging.info("Computing data for MFCC")

    y, X = phoneme_activations(audio_np, alignments, index=lambda ms: ms//10)
    result['mfcc'] = check_nan(features=X, labels=y)
    for name, net in nets:
        result[name] = {}
        index = lambda ms: encoders.inout(net.SpeechEncoder.Conv, torch.tensor(ms)//10).numpy()
        try:
            logging.info("Loading activations from activations.val.{}.npy".format(name))
            activations = np.load("activations.val.{}.npy".format(name), allow_pickle=True).item()
        except FileNotFoundError:    
            logging.info("Computing data for {}".format(name))
            activations = collect_activations(net, audio, batch_size=batch_size)
            activations['audio_id'] = np.array([ item['audio_id'] for item in phon ])
            logging.info("Saving activations to activations.val.{}.npy".format(name))
            np.save("activations.val.{}.npy".format(name), activations)
        for key in ['conv', 'rnn0', 'rnn1', 'rnn2', 'rnn3']:
                logging.info("Computing data for {}, {}".format(name, key))
                y, X = phoneme_activations(activations[key], alignments, index=index)
                result[name][key] = check_nan(features=X, labels=y)
    return result

def align2ipa(datum):
    """Extract IPA transcription from alignment information for a sentence."""
    from platalea.analysis.ipa import arpa2ipa
    result = []
    for word in datum['words']:
        for phoneme in word['phones']:
            result.append(arpa2ipa(phoneme['phone'].split('_')[0], '_'))
    return ''.join(result)
        
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
    

def phoneme_activations(activations, alignments, index=lambda ms: ms//10):
    """Return array of phoneme labels and array of corresponding mean-pooled activation states."""
    labels = []
    states = []
    for activation, alignment in zip(activations, alignments):
        # extract phoneme labels and activations for current utterance
        y, X = zip(*list(slices(alignment, activation, index=index)))
        y = np.array(y)
        X = np.stack(X)
        labels.append(y)
        states.append(X)
    return np.concatenate(labels), np.concatenate(states)


def logreg_acc(features, labels, test_size=1/3):
    """Fit logistic regression on part of features and labels and return accuracy on the other part."""
    #TODO tune penalty parameter C
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    scaler = StandardScaler() 
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=123)        
    X_train = scaler.fit_transform(X_train) 
    X_test  = scaler.transform(X_test) 
    m = LogisticRegression(penalty='l2', solver="lbfgs", multi_class='auto', max_iter=300, random_state=123, C=1.0) 
    m.fit(X_train, y_train) 
    return float(m.score(X_test, y_test)), m.coef_


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

def weight_variance():
    kinds = ['rnn0', 'rnn1', 'rnn2', 'rnn3']
    w = []
    layer = []
    trained = []
    for kind in kinds:
        rand = np.load("logreg_w_random_{}.npy".format(kind)).flatten()
        train = np.load("logreg_w_trained_{}.npy".format(kind)).flatten()
        w.append(rand)
        w.append(train)
        for _ in rand:
            layer.append(kind)
            trained.append('random')
        for _ in train:
            layer.append(kind)
            trained.append('trained')
        print(kind, "random", np.var(rand))
        print(kind, "trained", np.var(train))
    data = pd.DataFrame(dict(w = np.concatenate(w), layer=np.array(layer), trained=np.array(trained)))
    #g = ggplot(data, aes(y='w', x='layer')) + geom_violin() + facet_wrap('~trained', nrow=2, scales="free_y")
    g = ggplot(data, aes(y='w', x='layer')) + geom_point(position='jitter', alpha=0.1) + facet_wrap('~trained', nrow=2, scales="free_y") 
    ggsave(g, 'weight_variance.png')


    
def phoneme_rsa():
    logging.getLogger().setLevel('INFO')
    result = weighted_average_RSA(scalar=True, test_size=1/2, hidden_size=1024, epochs=60, device="cuda:0")
    json.dump(result, open('phoneme_rsa.json', 'w'), indent=2)

def phoneme_rsa_plot():
    data = pd.read_json("phoneme_rsa.json")
    order = ['mfcc', 'conv', 'rnn0', 'rnn1', 'rnn2', 'rnn3']
    data['layerid'] = [ order.index(x) for x in data['layer'] ] 
    g = ggplot(data, aes(x='layerid', y='cor', color='model')) + geom_point(size=2) + geom_line(size=2)
    ggsave(g, "phoneme_rsa.png")

def weighted_average_RSA(scalar=True, test_size=1/2, hidden_size=1024, epochs=1, device='cpu'):
    from sklearn.model_selection import train_test_split
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    splitseed = 123
    result = []
    logging.info("Loading transcription data")
    data = np.load("transcription.val.npy", allow_pickle=True)
    trans = [ datum['ipa'] for datum in data ]
    act = [ torch.tensor([item['audio'][:, :]]).float().to(device) for item in data ]

    trans, trans_val, act, act_val = train_test_split(trans, act, test_size=test_size, random_state=splitseed) 

    logging.info("Computing edit distances")
    edit_sim = torch.tensor(U.pairwise(S.stringsim, trans)).float().to(device)
    edit_sim_val = torch.tensor(U.pairwise(S.stringsim, trans_val)).float().to(device)
    logging.info("Training for input features")
    this = train_wa(edit_sim, edit_sim_val, act, act_val, scalar=scalar, hidden_size=hidden_size, epochs=epochs, device=device)
    result.append({**this, 'model': 'random', 'layer': 'mfcc'})
    result.append({**this, 'model': 'trained', 'layer': 'mfcc'})
    del act, act_val
    logging.info("Maximum correlation on val: {} at epoch {}".format(result[-1]['cor'], result[-1]['epoch']))
    for mode in ["random", "trained"]:
        logging.info("Loading activations for {} data".format(mode))
        data = np.load("activations.val.{}.npy".format(mode), allow_pickle=True).item()
        for layer in ['conv', 'rnn0', 'rnn1', 'rnn2', 'rnn2', 'rnn3']:
            logging.info("Training for {} {}".format(mode, layer))
            act = [ torch.tensor([item[:, :]]).float().to(device) for item in data[layer] ]
            act, act_val = train_test_split(act, test_size=test_size, random_state=splitseed)
            this = train_wa(edit_sim, edit_sim_val, act, act_val, scalar=scalar, hidden_size=hidden_size, epochs=epochs, device=device)
            result.append({**this, 'model': mode, 'layer': layer}) 
            del act, act_val
            print("Maximum correlation on val: {} at epoch {}".format(result[-1]['cor'], result[-1]['epoch']))
    return result

    
def train_wa(edit_sim, edit_sim_val, stack, stack_val, scalar=True, hidden_size=1024, epochs=1, device='cpu'):
    import platalea.encoders
    if scalar:
        wa = platalea.encoders.ScalarAttention(stack[0].size(2), hidden_size).to(device)
    else:
        # This crashes CUDA memory
        wa = platalea.encoders.Attention(stack[0].size(2), hidden_size).to(device)
    optim = torch.optim.Adam(wa.parameters())
    minloss = 0; minepoch = None
    logging.info("Optimizing for {} epochs".format(epochs))
    for epoch in range(1, 1+epochs):
        avg_pool = torch.cat([ wa(item) for item in stack])
        avg_pool_sim = S.cosine_matrix(avg_pool, avg_pool)
        loss = -S.pearson_r(S.triu(avg_pool_sim), S.triu(edit_sim))
        with torch.no_grad():
            avg_pool_val = torch.cat([ wa(item) for item in stack_val])
            avg_pool_sim_val = S.cosine_matrix(avg_pool_val, avg_pool_val)
            loss_val = -S.pearson_r(S.triu(avg_pool_sim_val), S.triu(edit_sim_val))
        logging.info("{} {} {}".format(epoch, -loss.item(), -loss_val.item()))
        if loss_val.item() <= minloss:
            minloss = loss_val.item()
            minepoch = epoch
        optim.zero_grad()
        loss.backward()
        optim.step()
        # Release CUDA-allocated tensors
        del loss, loss_val,  avg_pool, avg_pool_sim, avg_pool_val, avg_pool_sim_val
    del wa, optim
    return {'epoch': minepoch, 'cor': -minloss}
