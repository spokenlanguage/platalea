SEED=123
import torch
torch.manual_seed(SEED)
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
import sys
import torch.nn as nn
import platalea.basic as basic
import platalea.encoders as encoders
import platalea.attention
import os.path
import logging
import json

from plotnine import *
import pandas as pd 
import ursa.similarity as S
import ursa.util as U
import pickle




## Models            
### Local

def local_diagnostic():
    logging.getLogger().setLevel('INFO')
    output = []
    data_mfcc = pickle.load(open('local_input.pkl', 'rb'))
    for rep in ['trained', 'random']:
        data = pickle.load(open('local_{}.pkl'.format(rep), 'rb'))
        logging.info("Fitting Logistic Regression for mfcc")
        result  = logreg_acc_adam(data_mfcc['features'], data_mfcc['labels'], epochs=40, device='cuda:0')
        logging.info("Result for {}, {} = {}".format(rep, 'mfcc', result['acc']))
        #np.save('logreg_w_{}_{}.npy'.format(rep, 'mfcc'), w)
        result['model'] = rep
        result['layer'] = 'mfcc'
        output.append(result)
        for kind in data:
            logging.info("Fitting Logistic Regression for {}, {}".format(rep, kind))
            result = logreg_acc_adam(data[kind]['features'], data[kind]['labels'], epochs=40, device='cuda:0')
            logging.info("Result for {}, {} = {}".format(rep, kind, result['acc']))
            result['model'] = rep
            result['layer'] = kind
            output.append(result)
    json.dump(output, open("local_diagnostic.json", "w"), indent=True)

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
    result = weighted_average_diagnostic(attention='linear', test_size=1/2, hidden_size=1024, epochs=500, device="cuda:0")
    json.dump(result, open('global_diagnostic.json', 'w'), indent=2)

def plots():
    local_diagnostic_plot()
    global_diagnostic_plot()
    local_rsa_plot()
    global_rsa_plot()

## Plotting
def local_diagnostic_plot():
    data = pd.read_json("local_diagnostic.json", orient='records')
    order = ['mfcc', 'conv', 'rnn0', 'rnn1', 'rnn2', 'rnn3']
    data['layer_id'] = [ order.index(x) for x in data['layer'] ]
    data['rer'] = rer(data['acc'], data['baseline'])
    g = ggplot(data, aes(x='layer_id', y='rer', color='model')) + geom_point(size=2) + geom_line(size=2) + ylim(0, 1) + ggtitle("Local diagnostic")
    ggsave(g, 'local_diagnostic.png')

def global_diagnostic_plot():
    data = pd.read_json("global_diagnostic.json", orient='records')
    order = ['mfcc', 'conv', 'rnn0', 'rnn1', 'rnn2', 'rnn3']
    data['layer_id'] = [ order.index(x) for x in data['layer'] ]
    data['rer'] = rer(data['acc'], data['baseline'])
    g = ggplot(data, aes(x='layer_id', y='rer', color='model')) + geom_point(size=2) + geom_line(size=2) + ylim(0, 1) + ggtitle("Global diagnostic")
    ggsave(g, "global_diagnostic.png")

def local_rsa_plot():
    data = pd.read_json("local_rsa.json", orient='records')
    order = ['mfcc', 'conv', 'rnn0', 'rnn1', 'rnn2', 'rnn3']
    data['layer_id'] = [ order.index(x) for x in data['layer'] ] 
    g = ggplot(data, aes(x='layer_id', y='cor', color='model')) + geom_point(size=2) + geom_line(size=2) + ylim(0, 1) + ggtitle("Local RSA")
    ggsave(g, 'local_rsa.png')

def global_rsa_plot():
    data = pd.read_json("global_rsa.json", orient='records')
    order = ['mfcc', 'conv', 'rnn0', 'rnn1', 'rnn2', 'rnn3']
    data['layer_id'] = [ order.index(x) for x in data['layer'] ] 
    g = ggplot(data, aes(x='layer_id', y='cor', color='model')) + geom_point(size=2) + geom_line(size=2) + ylim(0, 1) + ggtitle("Global RSA")
    ggsave(g, "global_rsa.png")

## Tables
def learning_effect():
    def le(trained, random): 
        return 1 - (random/trained) 
    ld = pd.read_json("local_diagnostic.json", orient="records");    ld['scope'] = 'local';   ld['method'] = 'diagnostic'
    lr = pd.read_json("local_rsa.json", orient="records");           lr['scope'] = 'local';   lr['method'] = 'rsa'     
    gd = pd.read_json("global_diagnostic.json", orient="records");   gd['scope'] = 'global';  gd['method'] = 'diagnostic' 
    gr = pd.read_json("global_rsa.json", orient="records");          gr['scope'] = 'global';  gr['method'] = 'rsa'
    data = pd.concat([ld, lr, gd, gr], sort=False)
    data['rer'] = rer(data['acc'], data['baseline'])
    data['score'] = data['rer'].fillna(0.0) + data['cor'].fillna(0.0) 
    trained = data.loc[data['model']=='trained']
    random  = data.loc[data['model']=='random']
    trained['learning_effect'] = le(trained['score'].values, random['score'].values)
    data = trained[['epoch', 'layer', 'method', 'scope', 'score', 'learning_effect']].reset_index()
    order = ['mfcc', 'conv', 'rnn0', 'rnn1', 'rnn2', 'rnn3']
    data['layer_id'] = [ order.index(x) for x in data['layer'] ] 
    json.dump(data.to_dict(orient='records'), open('learning_effect.json', 'w'))
    g = ggplot(data, aes(x='layer_id', y='learning_effect', color='method', linetype='scope')) + geom_point(size=2) + geom_line(size=1) 
    ggsave(g, "learning_effect.png")

def majority_binary(y):
    return (y.mean(dim=0) >= 0.5).float()

def majority_multiclass(y):
    labels, counts = np.unique(y, return_counts=True)
    return labels[counts.argmax()]

def logreg_acc_adam(features, labels, test_size=1/2, epochs=1, device='cpu'):
    """Fit logistic regression on part of features and labels and return accuracy on the other part."""
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split

    X, X_val, y, y_val = train_test_split(features, labels, test_size=test_size, random_state=123)

    le = LabelEncoder()
    y = torch.tensor(le.fit_transform(y)).long()
    y_val = torch.tensor(le.transform(y_val)).long()
    
    scaler = StandardScaler() 
    X = torch.tensor(scaler.fit_transform(X)).float()
    X_val  = torch.tensor(scaler.transform(X_val)).float()

    model = SoftmaxClassifier(X.size(1), y.max().item()+1).to(device)
    result = train_classifier(model, X, y, X_val, y_val, epochs=epochs, majority=majority_multiclass)
    return result


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
    data = pickle.load(open("local_input.pkl", "rb"))
    for mode in ["trained", "random"]:
        mfcc_cor = [ item['cor']  for item in result if item['layer'] == 'mfcc']
        if len(mfcc_cor) > 0:
            logging.info("Result for MFCC computed previously")
            result.append(dict(model=mode, layer='mfcc', cor=mfcc_cor[0]))
        else:
            X, X_val, y, y_val = train_test_split(data['features'], data['labels'], test_size=test_size, random_state=splitseed)
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
        logging.info("Loading phoneme data for {}".format(mode))
        data = pickle.load(open("local_{}.pkl".format(mode), "rb"))
        for layer in ['conv', 'rnn0', 'rnn1', 'rnn2', 'rnn3']:
                X, X_val, y, y_val = train_test_split(data[layer]['features'], data[layer]['labels'], test_size=test_size, random_state=splitseed)
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

def framewise_RSA_no_matrix(test_size=1/2):
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
            cor = correlation_score(data['mfcc']['features'], data['mfcc']['labels'], size=test_size) 
            logging.info("Point biserial correlation for {}, mfcc: {}".format(mode, cor))
            result.append(dict(model=mode, layer='mfcc', cor=cor))
        for layer in ['conv', 'rnn0', 'rnn1', 'rnn2', 'rnn3']:
            cor = correlation_score(data[mode][layer]['features'], data[mode][layer]['labels'], size=test_size) 
            logging.info("Point biserial correlation for {}, {}: {}".format(mode, layer, cor))
            result.append(dict(model=mode, layer=layer, cor=cor))
    return result

def correlation_score(features, labels, size):
    from sklearn.metrics.pairwise import paired_cosine_distances
    from scipy.stats import pearsonr
    logging.info("Sampling 2x{} stimuli from a total of {}".format(size, len(labels)))
    indices = np.array(random.sample(range(len(labels)), size*2)) 
    y = labels[indices] 
    x = features[indices] 
    y_sim = y[: size] == y[size :] 
    x_sim = 1 - paired_cosine_distances(x[: size], x[size :]) 
    return pearsonr(x_sim, y_sim)[0]

    
def weighted_average_RSA(scalar=True, test_size=1/2, hidden_size=1024, epochs=1, device='cpu'):
    from sklearn.model_selection import train_test_split
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    splitseed = 123
    result = []
    logging.info("Loading transcription data")
    data = pickle.load(open("global_input.pkl", "rb"))
    trans = data['ipa']
    act = [ torch.tensor([item[:, :]]).float().to(device) for item in data['audio'] ]

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
        data = pickle.load(open("global_{}.pkl".format(mode), "rb"))
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
    if scalar:
        wa = platalea.attention.ScalarAttention(stack[0].size(2), hidden_size).to(device)
    else:
        # This crashes CUDA memory
        wa = platalea.attention.Attention(stack[0].size(2), hidden_size).to(device)
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

def weighted_average_diagnostic(attention='scalar', test_size=1/2, hidden_size=1024, epochs=1, device='cpu'):
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    splitseed = 123
    result = []
    logging.info("Loading transcription data")
    data = pickle.load(open("global_input.pkl", "rb"))
    trans = data['ipa']
    act = [ torch.tensor(item[:, :]).float().to(device) for item in data['audio'] ]
    
    trans, trans_val, X, X_val = train_test_split(trans, act, test_size=test_size, random_state=splitseed) 

    logging.info("Computing targets")
    vec = CountVectorizer(lowercase=False, analyzer='char')
    # Binary instead of counts
    y = torch.tensor(vec.fit_transform(trans).toarray()).float().clamp(min=0, max=1)
    y_val = torch.tensor(vec.transform(trans_val).toarray()).float().clamp(min=0, max=1)
    logging.info("Training for input features")
    #this = train_wa_diagnostic(X, y, X_val, y_val, attention=attention, hidden_size=hidden_size, epochs=epochs, device=device)
    model = PooledClassifier(input_size=X[0].shape[1], hidden_size=hidden_size, output_size=y[0].shape[0], attention=attention).to(device)
    this = train_classifier(model, X, y, X_val, y_val, epochs=epochs)
    result.append({**this, 'model': 'random', 'layer': 'mfcc'})
    result.append({**this, 'model': 'trained', 'layer': 'mfcc'})
    del X, X_val
    logging.info("Maximum accuracy on val: {} at epoch {}".format(result[-1]['acc'], result[-1]['epoch']))
    for mode in ["random", "trained"]:
        logging.info("Loading activations for {} data".format(mode))
        data = pickle.load(open("global_{}.pkl".format(mode), "rb"))
        for layer in ['conv', 'rnn0', 'rnn1', 'rnn2', 'rnn3']:
            logging.info("Training for {} {}".format(mode, layer))
            act = [ torch.tensor(item[:, :]).float() for item in data[layer] ]
            X, X_val = train_test_split(act, test_size=test_size, random_state=splitseed)
            
            #this = train_wa_diagnostic(X, y, X_val, y_val, attention=attention, hidden_size=hidden_size, epochs=epochs, device=device)
            model = PooledClassifier(input_size=X[0].shape[1], hidden_size=hidden_size, output_size=y[0].shape[0], attention=attention).to(device)
            this = train_classifier(model, X, y, X_val, y_val, epochs=epochs)
            result.append({**this, 'model': mode, 'layer': layer}) 
            del X, X_val
            print("Maximum accuracy on val: {} at epoch {}".format(result[-1]['acc'], result[-1]['epoch']))
    return result

# def train_wa_diagnostic(X, y, X_val, y_val, attention='scalar', hidden_size=1024, epochs=1, patience=50, device='cpu'):
#     import platalea.encoders
#     #model = PooledRegressor(X[0].size(1), hidden_size, y.size(1), attention=attention).to(device)
#     model = PooledClassifier(X[0].size(1), hidden_size, y.size(1), attention=attention).to(device)
#     optim = torch.optim.Adam(model.parameters(), lr=1e-3)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', factor=0.1, patience=10)
#     minloss = np.finfo(np.float32).max; minepoch = 0; minacc = 0
#     data = torch.utils.data.DataLoader(list(zip(X, y)), batch_size=64, shuffle=True, collate_fn=collate)
#     data_val = torch.utils.data.DataLoader(list(zip(X_val, y_val)), batch_size=64, shuffle=False, collate_fn=collate)
#     logging.info("Optimizing for {} epochs".format(epochs))
#     N_val = y_val.shape[0] * y_val.shape[1]
#     with torch.no_grad():
#         mean = (y.mean(dim=0) >= 0.0).float()
#         base = np.sum([(mean.expand_as(y_i) == y_i).sum().item() for y_i in y]) / N_val
#         logging.info("Baseline accuracy: {}".format(base))
#     for epoch in range(1, 1+epochs):
#         epoch_loss = []
#         for x, y in data:
#             x = x.to(device)
#             y = y.to(device)
#             y_pred = model(x) 
#             #loss = nn.functional.mse_loss(y_pred, y)
#             loss = nn.functional.binary_cross_entropy_with_logits(y_pred, y)
#             optim.zero_grad()
#             loss.backward()
#             optim.step()
#             epoch_loss.append(loss.item())
#         with torch.no_grad():
#             #loss_val = np.sum([nn.functional.mse_loss(model(x.to(device)), y.to(device), reduction='sum').item() for x, y in data_val]) / N_val
#             loss_val = np.sum([nn.functional.binary_cross_entropy_with_logits(model(x.to(device)), y.to(device), reduction='sum').item() for x, y in data_val]) / N_val
#             accuracy_val = np.sum([(model.predict(x.to(device)) == y.to(device)).sum().item() for x, y in data_val]) / N_val
#             scheduler.step(loss_val)
#             logging.info("{} {} {} {}".format(epoch, np.mean(epoch_loss), loss_val, accuracy_val))
#         if loss_val <= minloss:
#             minloss = loss_val; minepoch = epoch; minacc = accuracy_val
#         if epoch - minepoch >= patience:
#             logging.info("No improvement for {} epochs, stopping.".format(patience))
#             break
#         # Release CUDA-allocated tensors
#         del x, y, loss, loss_val,  y_pred
#     del model, optim
#     return {'epoch': minepoch, 'error': minloss, 'acc': minacc, 'base': base}

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

class PooledClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, attention='scalar'):
        super(PooledClassifier, self).__init__()
        if attention == 'scalar':
            self.wa = platalea.attention.ScalarAttention(input_size, hidden_size)
        elif attention == 'linear':
            self.wa = platalea.attention.LinearAttention(input_size)
        else:
            self.wa = platalea.attention.Attention(input_size, hidden_size)
        self.project = nn.Linear(in_features=input_size, out_features=output_size)
        self.loss = nn.functional.binary_cross_entropy_with_logits
        
    def forward(self, x):
        return self.project(self.wa(x))
    
    def predict(self, x):
        logit = self.project(self.wa(x))
        return (logit >= 0.0).float()


class SoftmaxClassifier(nn.Module):

    def __init__(self, input_size, output_size):
        super(SoftmaxClassifier, self).__init__()
        self.project = nn.Linear(in_features=input_size, out_features=output_size)
        self.loss = nn.functional.cross_entropy
        
    def forward(self, x):
        return self.project(x)

    def predict(self, x):
        return self.project(x).argmax(dim=1)

    
    
def collate(items):
    x, y = zip(*items)
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0)
    y = torch.stack(y)
    return x, y

def tuple_stack(xy):
    x, y = zip(*xy)
    return torch.stack(x), torch.stack(y)

def rer(hi, lo): 
    return ((1-lo) - (1-hi))/(1-lo)

def train_classifier(model, X, y, X_val, y_val, epochs=1, patience=50, majority=majority_binary):
    device = list(model.parameters())[0].device
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', factor=0.1, patience=10)
    data = torch.utils.data.DataLoader(list(zip(X, y)), batch_size=64, shuffle=True, collate_fn=collate)
    data_val = torch.utils.data.DataLoader(list(zip(X_val, y_val)), batch_size=64, shuffle=False, collate_fn=collate)
    logging.info("Optimizing for {} epochs".format(epochs))
    scores = []
    with torch.no_grad():
        maj = majority(y)
        baseline = np.mean([ (maj == y_i).cpu().numpy() for y_i in y_val ])
        logging.info("Baseline accuracy: {}".format(baseline))
    for epoch in range(1, 1+epochs):
        epoch_loss = []
        for x, y in data:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x) 
            loss = model.loss(y_pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss.append(loss.item())
        with torch.no_grad():
            loss_val = np.mean( [model.loss(model(x.to(device)), y.to(device)).item() for x,y in data_val])
            accuracy_val = np.concatenate([ model.predict(x.to(device)).cpu().numpy() == y.cpu().numpy() for x, y in data_val]).mean()
            scheduler.step(loss_val)
            logging.info("{} {} {} {}".format(epoch, np.mean(epoch_loss), loss_val, accuracy_val))
        scores.append(dict(epoch=epoch, train_loss=np.mean(epoch_loss), acc=accuracy_val, loss=loss_val, baseline=baseline))
        minepoch = min(scores, key=lambda a: a['loss'])['epoch']
        if epoch - minepoch >= patience:
            logging.info("No improvement for {} epochs, stopping.".format(patience))
            break
        # Release CUDA-allocated tensors
        del x, y, loss, loss_val,  y_pred
    del model, optim
    return min(scores, key=lambda a: a['loss'])

