import json
import pandas as pd
import glob
import io
import os.path
import logging
logging.basicConfig(level=logging.INFO)

def load_results(d, fname='result.json'):
    return [ json.loads(line) for line in open("{}/{}".format(d, fname)) ]

def select(path, spec):
    data = json.load(open(path))
    return [ x for x in data if all(x[key] == val for key, val in spec.items()) ][0]

def scores():
    for d in glob.glob("experiments/vq-*"):
        try:
            logging.info("Loading results from {}".format(d))
            size = d.split('-')[1]
            level = d.split('-')[2][1]
            
            ret = sorted(load_results(d), key=lambda x: x['recall']['10'])[-1]
            ret_base = load_results(d)[0]
            zs  = json.load(open("{}/vq_result.json".format(d)))
            zs_base = json.load(open("{}/vq_base_result.json".format(d)))
            
            
            cor      = select("{}/ed_rsa.json".format(d), dict(model='trained', reference='phoneme'))['cor']
            cor_base = select("{}/ed_rsa.json".format(d), dict(model='random', reference='phoneme'))['cor']
            cor_word = select("{}/ed_rsa.json".format(d), dict(model='trained', reference='word'))['cor']
            cor_word_base = select("{}/ed_rsa.json".format(d), dict(model='random', reference='word'))['cor']
            diag     = select("{}/local/local_diagnostic.json".format(d), dict(model='trained'))['acc']
            diag_base = select("{}/local/local_diagnostic.json".format(d), dict(model='random'))['acc']
        
            trained = dict(
                condition=os.path.basename(d),
                mode = 'trained',
                size=size,
                level=level,
                epoch=ret['epoch'],
                recall=ret['recall']['10'],
                abx=100-zs['2019']['english']['scores']['abx'],
                abx_lev=100-zs['2019']['english']['details_abx']['test']['levenshtein'],
                bitrate=zs['2019']['english']['scores']['bitrate'],
                ed_rsa=cor,
                ed_rsa_word=cor_word,
                recall_diff=ret['recall']['10'] - ret_base['recall']['10'],
                abx_diff=(100-zs['2019']['english']['scores']['abx']) - (100-zs_base['2019']['english']['scores']['abx']),
                ed_rsa_diff=cor - cor_base ,
                ed_rsa_word_diff=cor_word - cor_word_base,
                diag=diag,
                diag_diff=diag-diag_base)
        except FileNotFoundError as e:
            logging.warning("MISSING DATA FOR {}\\{}".format(d,e))
            
        yield trained

        
        
data = pd.read_json(json.dumps(list(scores())), orient='records')

print(data)
from plotnine import *

vars = ['abx', 'abx_lev', 'ed_rsa', 'ed_rsa_word', 'diag']
for var in vars:
    
    p = ggplot(data, aes(x='recall', y=var)) + \
        geom_point(aes(size='bitrate', shape='factor(level)', color='factor(size)'))
    ggsave(p, 'plot-recall-{}.pdf'.format(var))

    for var2 in vars:
        if var < var2:
            p = ggplot(data, aes(x=var, y=var2)) + \
                geom_point(aes(size='bitrate', shape='factor(level)', color='factor(size)'))
            ggsave(p, 'plot-{}-{}.pdf'.format(var, var2))
