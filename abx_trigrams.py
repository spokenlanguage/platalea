import pydub
import os.path
from pathlib import Path
import json
import platalea.ipa as ipa
import h5py
import logging
import zerospeech2020.evaluation.abx as abx
from ABXpy.misc.any2h5features import convert
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import shutil
import tempfile
import ABXpy
import ABXpy.task
from zerospeech2020.evaluation.abx import _load_features_2019
import ABXpy.distances.distances
from ABXpy.distance import edit_distance
import ABXpy.score as score
import ABXpy.misc.items as items
import ABXpy.analyze as analyze
import pandas as pd

def phonemes(u):
    result = []
    for word in u['words']:
        if word['case'] == 'success':
            current = word['start']
            for phone in word['phones']:
                phone['start'] = current
                phone['end'] = current + phone['duration']
                current = phone['end']
                result.append(phone)
    return result
    
def trigrams(xs):
    if len(xs) < 3:
        return []
    else:
        return [xs[0:3]] + trigrams(xs[1:])


def deoov(xs):
    return [ x for x in xs if not any(xi['phone'].startswith('oov') or xi['phone'].startswith('sil') for xi in x) ]


class Topline:

    def __init__(self):
        self.oh = OneHotEncoder(dtype=np.int32, sparse=False)
        #phonemes = np.array(list(ipa._arpa2ipa.values())).reshape(-1, 1)
        phonemes = np.array(list(ipa._arpa2ipa.keys())).reshape(-1, 1)
        self.oh.fit(phonemes)

    def transform(self, xs):
        features = np.array(xs).reshape(-1,1)
        return self.oh.transform(features)

    def dump(self, xs, path):
        np.savetxt(path, self.transform(xs))
            
def prepare_abx(us):
    import csv
    wav = Path("data/datasets/flickr8k/flickr_audio/wavs")
    out = Path("data/flickr8k_abx_wav")
    out.mkdir(parents=True, exist_ok=True)
    items = csv.writer(open("data/flickr8k_abx.item", "w"), delimiter=' ', lineterminator='\n')
    items.writerow(["#file", "onset", "offset", "#middle", "trigram", "context"])
    for u in us:        
        filename = os.path.split(u['audiopath'])[-1]
        bare, _ = os.path.splitext(filename)
        grams = deoov(trigrams(phonemes(u)))
        logging.info("Loading audio from {}".format(filename))
        sound = pydub.AudioSegment.from_file(wav / filename)
        Path(out / bare).mkdir(parents=True, exist_ok=True)
        for i, gram in enumerate(grams):
            start = int(gram[0]['start']*1000)
            end = int(gram[-1]['end']*1000)
            triple = [ phone['phone'].split('_')[0] for phone in gram ]
            fragment = sound[start : end]
            target = out / "{}_{}.wav".format(bare, i)
            items.writerow([ "{}_{}".format(bare,i), 0, end-start, triple[1], '_'.join(triple), '_'.join([triple[0], triple[-1]])])
            fragment.export(format='wav', out_f=target)
            logging.info("Saved {}th trigram in {}".format(i, target))


def ed(x, y, normalized=None): 
    return edit_distance(x, y) 

def run_abx(feature_dir, item_file):
    temp = Path(tempfile.mkdtemp())
    try:
        logging.info("Converting features")
        convert(feature_dir, temp / "features", load=_load_features_2019) 
        task = ABXpy.task.Task(item_file, "middle", by="context")
        logging.info("Generating triplets")
        task.generate_triplets(output= str(temp / "triplets"))
        logging.info("Computing distances")
        ABXpy.distances.distances.compute_distances( 
            str(temp / "features"),
            'features', 
            str(temp / "triplets"),
            str(temp / "distance"),
            ed,
            normalized=True,
            n_cpu=16)
        logging.info("Computing scores")
        score.score(str(temp / "triplets"), str(temp / "distance"), str(temp / "score"))
        analyze.analyze(str(temp / "triplets"), str(temp / "score"), str(temp / "analyze"))
        data = pd.read_csv(temp / "analyze", delimiter='\t')
        return data
    finally:
        shutil.rmtree(temp)
        



def main():
    logging.basicConfig(level=logging.INFO)
    align = {}
    for line in open("data/datasets/flickr8k/fa.json"):
        u = json.loads(line)
        u['audiopath'] =  os.path.split(u['audiopath'])[-1]
        align[u['audiopath']] = u
    prepare_abx(list(align.values())[:1000])
    


if __name__ == '__main__':
    main()
    

