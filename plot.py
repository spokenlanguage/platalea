import json
import pandas as pd
import glob
import io
import os.path

def load_results(d):
    return [ json.loads(line) for line in open("{}/result.json".format(d)) ]

def scores():
    for d in glob.glob("experiments/vq-*"):
        size = d.split('-')[1]
        level = d.split('-')[2][1]
        ret = sorted(load_results(d), key=lambda x: x['recall']['10'])[-1]
        zs  = json.load(open("{}/vq_result.json".format(d)))
        yield dict(
            condition=os.path.basename(d),
            size=size,
            level=level,
            epoch=ret['epoch'],
            recall=ret['recall']['10'],
            abx=zs['2019']['english']['scores']['abx'],
            bitrate=zs['2019']['english']['scores']['bitrate'])


data = pd.read_json(json.dumps(list(scores())), orient='records')

from plotnine import *

p = ggplot(data, aes(x='recall', y='100-abx')) + \
    geom_point(aes(size='bitrate', shape='factor(level)', color='factor(size)')) + \
    ylab('ABX accuracy') + \
    xlab("Image retrieval recall @ 10")
ggsave(p, 'plot.pdf')

