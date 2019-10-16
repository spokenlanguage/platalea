import sys
import torch
torch.manual_seed(123)
import torch.nn as nn
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
import pickle


def prepare():
    logging.getLogger().setLevel('INFO')
    logging.info("Loading pytorch models")
    net_rand = basic.SpeechImage(basic.DEFAULT_CONFIG).cuda()
    net_train = basic.SpeechImage(basic.DEFAULT_CONFIG)
    net_train.load_state_dict(torch.load("experiments/basic-stack/net.20.pt").state_dict())
    net_train.cuda()
    nets = [('trained', net_train), ('random', net_rand)]
    # Recurrent
    for rep, net in nets:
        with torch.no_grad():
            net.eval()
            data = phoneme_data([(rep, net)], batch_size=32, framewise=True) 
            pickle.dump(data, open('phoneme_data_{}.pkl'.format(rep), 'wb'), protocol=4)    

## Models            
### Local

def local_diagnostic():
    result = []
    for rep in ['trained', 'random']:
        data = pickle.load(open('phoneme_data_{}.pkl'.format(rep), 'rb'))
        logging.info("Fitting Logistic Regression for mfcc")
        acc  = logreg_acc_adam(data['mfcc']['features'], data['mfcc']['labels'], epochs=40, device='cuda:0')
        logging.info("Result for {}, {} = {}".format(rep, 'mfcc', acc))
        #np.save('logreg_w_{}_{}.npy'.format(rep, 'mfcc'), w)
        result.append(dict(model=rep, layer='mfcc', layer_id=0, acc=acc))
        for j, kind in enumerate(data[rep], start=1):
                logging.info("Fitting Logistic Regression for {}, {}".format(rep, kind))
                acc = logreg_acc_adam(data[rep][kind]['features'], data[rep][kind]['labels'], epochs=40, device='cuda:0')
                logging.info("Result for {}, {} = {}".format(rep, kind, acc))
                #np.save('logreg_w_{}_{}.npy'.format(rep, kind), w)
                result.append(dict(model=rep, layer=kind, layer_id=j, acc=acc))
    json.dump(result, open("local_diagnostic_framewise.json", "w"), indent=True)

def local_rsa():
    logging.getLogger().setLevel('INFO')
    result = framewise_RSA(test_size=70000)
    json.dump(result, open('local_rsa.json', 'w'), indent=2)
    
### Global
    
def global_rsa():
    logging.getLogger().setLevel('INFO')
    result = weighted_average_RSA(scalar=True, test_size=1/2, hidden_size=1024, epochs=60, device="cuda:0")
    #result = weighted_average_RSA(scalar=False, test_size=1/2, hidden_size=1024, epochs=60, device="cpu")
    json.dump(result, open('global_rsa.json', 'w'), indent=2)

def global_diagnostic():
    logging.getLogger().setLevel('INFO')
    result = weighted_average_diagnostic(scalar=True, test_size=1/2, hidden_size=1024, epochs=100, device="cuda:0")
    json.dump(result, open('global_diagnostic_scalar.json', 'w'), indent=2)


## Plotting
def local_diagnostic_plot():
    data = pd.read_json("local_diagnostic_framewise.json", orient='records')
    order = ['mfcc', 'conv', 'rnn0', 'rnn1', 'rnn2', 'rnn3']
    data['layer_id'] = [ order.index(x) for x in data['layer'] ] 
    g = ggplot(data, aes(x='layer_id', y='acc', color='model')) + geom_point(size=2) + geom_line(size=2) + ylim(0, max(data['acc']))
    ggsave(g, 'local_diagnostic_framewise.png')

def local_rsa_plot():
    data = pd.read_json("local_rsa.json", orient='records')
    order = ['mfcc', 'conv', 'rnn0', 'rnn1', 'rnn2', 'rnn3']
    data['layer_id'] = [ order.index(x) for x in data['layer'] ] 
    g = ggplot(data, aes(x='layer_id', y='cor', color='model')) + geom_point(size=2) + geom_line(size=2) + ylim(0, max(data['cor']))
    ggsave(g, 'local_rsa.png')

def global_rsa_plot():
    data = pd.read_json("global_rsa.json")
    order = ['mfcc', 'conv', 'rnn0', 'rnn1', 'rnn2', 'rnn3']
    data['layer_id'] = [ order.index(x) for x in data['layer'] ] 
    g = ggplot(data, aes(x='layer_id', y='cor', color='model')) + geom_point(size=2) + geom_line(size=2) + ylim(0, max(data['cor']))
    ggsave(g, "global_rsa.png")

def global_diagnostic_plot():
    data = pd.read_json("global_diagnostic.json")
    order = ['mfcc', 'conv', 'rnn0', 'rnn1', 'rnn2', 'rnn3']
    data['layer_id'] = [ order.index(x) for x in data['layer'] ] 
    g = ggplot(data, aes(x='layer_id', y='error', color='model')) + geom_point(size=2) + geom_line(size=2)
    ggsave(g, "global_diagnostic.png")

def phoneme_data(nets,  batch_size, framewise=True):
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

    y, X = phoneme_activations(audio_np, alignments, index=lambda ms: ms//10, framewise=framewise)
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
    return float(m.score(X_test, y_test))

def logreg_acc_adam(features, labels, test_size=1/3, epochs=1, device='cpu'):
    """Fit logistic regression on part of features and labels and return accuracy on the other part."""
    #TODO tune penalty parameter C
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=test_size, random_state=123)

    le = LabelEncoder()
    y_train = torch.tensor(le.fit_transform(y_train)).long()
    y_val = torch.tensor(le.transform(y_val)).long()
    
    scaler = StandardScaler() 
    X_train = torch.tensor(scaler.fit_transform(X_train)).float()
    X_val  = torch.tensor(scaler.transform(X_val)).float()

    result = train_diagnostic(X_train, y_train, X_val, y_val, epochs=epochs, device=device)
    return result['accuracy']

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


def framewise_RSA(test_size=1/2):
    from sklearn.model_selection import train_test_split
    #from sklearn.metrics.pairwise import cosine_similarity
    splitseed = 123
    result = []
    mfcc_done = False
    for mode in ["trained", "random"]:
        logging.info("Loading phoneme data for {}".format(mode))
        data = pickle.load(open("phoneme_data_{}.pkl".format(mode), "rb"))
        mfcc_cor = [ item['cor']  for item in result if item['layer'] == 'mfcc']
        if len(mfcc_cor) > 0:
            logging.info("Result for MFCC computed previously")
            result.append(dict(model=mode, layer='mfcc', cor=mfcc_cor[0]))
        else:
            X, X_val, y, y_val = train_test_split(data['mfcc']['features'], data['mfcc']['labels'], test_size=test_size, random_state=splitseed)
            logging.info("Computing label identity matrix for {} datapoints".format(len(y_val)))
            # y_sim = y.reshape((-1, 1)) == y
            y_val_sim = torch.tensor(y_val.reshape((-1, 1)) == y_val).float()
            logging.info("Computing activation similarities for {} datapoints".format(len(X_val)))
            # X_sim = S.cosine_matrix(X, X)
            X_val = torch.tensor(X_val).float()
            X_val_sim = S.cosine_matrix(X_val, X_val)
            cor = S.pearson_r(S.triu(y_val_sim), S.triu(X_val_sim)).item()
            logging.info("Point biserial correlation for {}, mfcc: {}".format(mode, cor))
            result.append(dict(model=mode, layer='mfcc', cor=cor))
        for layer in ['conv', 'rnn0', 'rnn1', 'rnn2', 'rnn3']:
                X, X_val, y, y_val = train_test_split(data[mode][layer]['features'], data[mode][layer]['labels'], test_size=test_size, random_state=splitseed)
                logging.info("Computing label identity matrix for {} datapoints".format(len(y_val)))
                # y_sim = y.reshape((-1, 1)) == y
                y_val_sim = torch.tensor(y_val.reshape((-1, 1)) == y_val).float()
                logging.info("Computing activation similarities for {} datapoints".format(len(X_val)))
                # X_sim = S.cosine_matrix(X, X)
                X_val = torch.tensor(X_val).float()
                X_val_sim = S.cosine_matrix(X_val, X_val)
                cor = S.pearson_r(S.triu(y_val_sim), S.triu(X_val_sim)).item()
                logging.info("Point biserial correlation for {}, {}: {}".format(mode, layer, cor))
                result.append(dict(model=mode, layer=layer, cor=cor))
    return result

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
        for layer in ['conv', 'rnn0', 'rnn1', 'rnn2', 'rnn3']:
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

def weighted_average_diagnostic(scalar=True, test_size=1/2, hidden_size=1024, epochs=1, device='cpu'):
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    splitseed = 123
    result = []
    logging.info("Loading transcription data")
    data = np.load("transcription.val.npy", allow_pickle=True)
    trans = [ datum['ipa'] for datum in data ]
    act = [ torch.tensor(item['audio'][:, :]).float() for item in data ]

    trans, trans_val, X, X_val = train_test_split(trans, act, test_size=test_size, random_state=splitseed) 

    logging.info("Computing targets")
    vec = CountVectorizer(lowercase=False, analyzer='char')
    y = torch.tensor(vec.fit_transform(trans).toarray()).float()
    y_val = torch.tensor(vec.transform(trans_val).toarray()).float()
    logging.info("Training for input features")
    this = train_wa_diagnostic(X, y, X_val, y_val, scalar=scalar, hidden_size=hidden_size, epochs=epochs, device=device)
    result.append({**this, 'model': 'random', 'layer': 'mfcc'})
    result.append({**this, 'model': 'trained', 'layer': 'mfcc'})
    del X, X_val
    logging.info("Maximum r2 on val: {} at epoch {}".format(result[-1]['r2'], result[-1]['epoch']))
    for mode in ["random", "trained"]:
        logging.info("Loading activations for {} data".format(mode))
        data = np.load("activations.val.{}.npy".format(mode), allow_pickle=True).item()
        for layer in ['conv', 'rnn0', 'rnn1', 'rnn2', 'rnn2', 'rnn3']:
            logging.info("Training for {} {}".format(mode, layer))
            act = [ torch.tensor(item[:, :]).float() for item in data[layer] ]
            X, X_val = train_test_split(act, test_size=test_size, random_state=splitseed)
            this = train_wa_diagnostic(X, y, X_val, y_val, scalar=scalar, hidden_size=hidden_size, epochs=epochs, device=device)
            result.append({**this, 'model': mode, 'layer': layer}) 
            del X, X_val
            print("Maximum r2 on val: {} at epoch {}".format(result[-1]['r2'], result[-1]['epoch']))
    return result

def train_wa_diagnostic(X, y, X_val, y_val, scalar=True, hidden_size=1024, epochs=1, device='cpu'):
    import platalea.encoders
    model = PooledRegressor(X[0].size(1), hidden_size, y.size(1), scalar=scalar).to(device)
    optim = torch.optim.Adam(model.parameters())
    minloss = np.finfo(np.float32).max; minepoch = None
    data = torch.utils.data.DataLoader(list(zip(X, y)), batch_size=128, shuffle=True, collate_fn=collate)
    data_val = torch.utils.data.DataLoader(list(zip(X_val, y_val)), batch_size=128, shuffle=False, collate_fn=collate)
    logging.info("Optimizing for {} epochs".format(epochs))
    N_val = y_val.shape[0] * y_val.shape[1]
    with torch.no_grad():
        mean = y_val.mean(dim=0)
        mse_base = np.sum([nn.functional.mse_loss(mean.expand_as(y_i), y_i, reduction='sum').item() for y_i in y]) / N_val
    for epoch in range(1, 1+epochs):
        epoch_loss = []
        for x, y in data:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x) 
            loss = nn.functional.mse_loss(y_pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss.append(loss.item())
        with torch.no_grad():
            loss_val = np.sum([nn.functional.mse_loss(model(x.to(device)), y.to(device), reduction='sum').item() for x, y in data_val]) / N_val
            logging.info("{} {} {} {}".format(epoch, np.mean(epoch_loss), loss_val, 1 - loss_val / mse_base))
        if loss_val <= minloss:
            minloss = loss_val
            minepoch = epoch
        # Release CUDA-allocated tensors
        del loss, loss_val,  y_pred
    del model, optim
    return {'epoch': minepoch, 'error': minloss, 'r2': 1 - minloss / mse_base}

def train_diagnostic(X, y, X_val, y_val, epochs=1, device='cpu'):
    from sklearn.metrics import accuracy_score
    model = nn.Linear(X.size(1), y.max().item()+1).to(device)
    optim = torch.optim.Adam(model.parameters())
    minloss = np.finfo(np.float32).max; minepoch = None
    data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=128, shuffle=True, pin_memory=True)
    data_val = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_val, y_val), batch_size=128, shuffle=False, pin_memory=True)
    logging.info("Optimizing for {} epochs".format(epochs))
    N_val = y_val.shape[0] # FIXME this needs checked
    for epoch in range(1, 1+epochs):
        epoch_loss = []
        for x, y in data:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = nn.functional.cross_entropy(y_pred, y)
            #print(torch.stack([y_pred.argmax(dim=1), y]).t())
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss.append(loss.item())
        with torch.no_grad():
            loss_val = torch.sum([nn.functional.cross_entropy(model(x.to(device)), y.to(device), reduction='sum').item() for x, y in data_val]) / N_val
            logging.info("{} {} {}".format(epoch, np.mean(epoch_loss), loss_val))
        if loss_val <= minloss:
            minloss = loss_val
            minepoch = epoch

        # Release CUDA-allocated tensors
        #del loss, loss_val,  y_pred
    # compute accuracy
    y_val_pred = torch.cat([model(x.to(device)) for x, y in data_val]).argmax(dim=1)
    acc = accuracy_score(y_val.detach().cpu().numpy(), y_val_pred.detach().cpu().numpy())
    del model, optim
    return {'epoch': minepoch, 'error': minloss, 'accuracy': acc}

class PooledRegressor(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, scalar=True):
        super(PooledRegressor, self).__init__()
        if scalar:
            self.wa = encoders.ScalarAttention(input_size, hidden_size)
        else:
            self.wa = encoders.Attention(input_size, hidden_size)

        self.project = nn.Linear(in_features=input_size, out_features=output_size)

    def forward(self, x):
        return nn.functional.relu(self.project(self.wa(x)))

class SoftmaxRegressor(nn.Module):

    def __init__(self, input_size, output_size):
        super(SoftmaxRegressor, self).__init__()
        self.project = nn.Linear(in_features=input_size, out_features=output_size)

    def forward(self, x):
        return nn.functional.softmax(self.project(x))
    
def collate(items):
    x, y = zip(*items)
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0)
    y = torch.stack(y)
    return x, y

def tuple_stack(xy):
    x, y = zip(*xy)
    return torch.stack(x), torch.stack(y)
