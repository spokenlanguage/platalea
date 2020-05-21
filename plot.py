import json
import pandas as pd
import glob
import io
import os.path
import logging
logging.basicConfig(level=logging.INFO)

def load_results(d, fname='result.json'):
    return [ json.loads(line) for line in open("{}/{}".format(d, fname)) ]

def scores():
    for d in glob.glob("experiments/vq-*"):
        logging.info("Loading results from {}".format(d))
        size = d.split('-')[1]
        level = d.split('-')[2][1]
        ret = sorted(load_results(d), key=lambda x: x['recall']['10'])[-1]
        ret_base = load_results(d)[0]
        zs  = json.load(open("{}/vq_result.json".format(d)))
        try:
            zs_base = json.load(open("{}/vq_base_result.json".format(d)))
        except FileNotFoundError:
            # FIXME FAKE!!!
            zs_base = {'2019': {'english': {'scores': {'abx': 100, 'bitrate': 1000}}}}
            
        cor  = [ x for x in json.load(open("{}/ed_rsa.json".format(d)))
                 if x['model'] == 'trained' ][0]['cor']
        cor_base = [ x for x in json.load(open("{}/ed_rsa.json".format(d)))
                     if x['model'] == 'random' ][0]['cor']
        trained = dict(
            condition=os.path.basename(d),
            mode = 'trained',
            size=size,
            level=level,
            epoch=ret['epoch'],
            recall=ret['recall']['10'],
            abx=100-zs['2019']['english']['scores']['abx'],
            bitrate=zs['2019']['english']['scores']['bitrate'],
            ed_rsa=cor,
            recall_diff=ret['recall']['10'] - ret_base['recall']['10'],
            abx_diff=(100-zs['2019']['english']['scores']['abx']) - (100-zs_base['2019']['english']['scores']['abx']),
            bitrate_diff=zs['2019']['english']['scores']['bitrate'] - zs_base['2019']['english']['scores']['bitrate'],
            ed_rsa_diff=cor - cor_base )
            
        yield trained

        
        
data = pd.read_json(json.dumps(list(scores())), orient='records')

print(data)
from plotnine import *

p = ggplot(data, aes(x='recall', y='abx')) + \
    geom_point(aes(size='bitrate', shape='factor(level)', color='factor(size)')) + \
    ylab('ABX accuracy') + \
    xlab("Image retrieval recall @ 10")
ggsave(p, 'plot-recall-abx.pdf')

p = ggplot(data, aes(x='recall', y='ed_rsa')) + \
    geom_point(aes(size='bitrate', shape='factor(level)', color='factor(size)')) + \
    ylab('RSA with phonemes') + \
    xlab("Image retrieval recall @ 10")
ggsave(p, 'plot-recall-rsa.pdf')

p = ggplot(data, aes(x='abx', y='ed_rsa')) + \
    geom_point(aes(size='bitrate', shape='factor(level)', color='factor(size)')) + \
    ylab('RSA with phonemes') + \
    xlab("ABX accuracy")
ggsave(p, 'plot-abx-rsa.pdf')

p = ggplot(data, aes(x='abx_diff', y='ed_rsa_diff')) + \
    geom_point(aes(size='bitrate', shape='factor(level)', color='factor(size)')) + \
    ylab('RSA with phonemes above random') + \
    xlab("ABX accuracy above random")
ggsave(p, 'plot-abx-rsa_diff.pdf')
