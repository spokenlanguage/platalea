import pydub
import os
import os.path
from pathlib import Path
import json
import platalea.ipa as ipa
import logging
from ABXpy.misc.any2h5features import convert
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import shutil
import ABXpy
import ABXpy.task
from zerospeech2020.evaluation.abx import _load_features_2019, _average
import ABXpy.distances.distances
from ABXpy.distance import edit_distance
import ABXpy.score as score
import ABXpy.analyze as analyze
import pandas as pd
import torch
import platalea.basicvq as M
import random


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
    return [x for x in xs if not any(xi['phone'].startswith('oov') or xi['phone'].startswith('sil') for xi in x)]


class Topline:

    def __init__(self):
        self.oh = OneHotEncoder(dtype=np.int32, sparse=False)
        #phonemes = np.array(list(ipa._arpa2ipa.values())).reshape(-1, 1)
        phonemes = np.array(list(ipa._arpa2ipa.keys())).reshape(-1, 1)
        self.oh.fit(phonemes)

    def transform(self, xs):
        features = np.array(xs).reshape(-1, 1)
        return self.oh.transform(features)

    def dump(self, xs, path):
        np.savetxt(path, self.transform(xs))


def prepare_abx(k=1000):
    import csv
    align = {}
    for line in open("data/datasets/flickr8k/fa.json"):
        u = json.loads(line)
        u['audiopath'] = os.path.split(u['audiopath'])[-1]
        align[u['audiopath']] = u
    us = random.sample(list(align.values()), k)
    wav = Path("data/datasets/flickr8k/flickr_audio/wavs")
    out = Path("data/flickr8k_abx_wav")
    speakers = dict(line.split() for line in open("data/datasets/flickr8k/wav2spk.txt"))
    out.mkdir(parents=True, exist_ok=True)
    with open("data/flickr8k_abx.item", "w") as itemout:
        items = csv.writer(itemout, delimiter=' ', lineterminator='\n')
        items.writerow(["#file", "onset", "offset", "#phone", "speaker", "context"])
        for u in us:
            filename = os.path.split(u['audiopath'])[-1]
            speaker = speakers[filename]
            bare, _ = os.path.splitext(filename)
            grams = deoov(trigrams(phonemes(u)))
            logging.info("Loading audio from {}".format(filename))
            sound = pydub.AudioSegment.from_file(wav / filename)
            Path(out / bare).mkdir(parents=True, exist_ok=True)
            for i, gram in enumerate(grams):
                start = int(gram[0]['start']*1000)
                end = int(gram[-1]['end']*1000)
                triple = [phone['phone'].split('_')[0] for phone in gram]
                fragment = sound[start: end]
                target = out / "{}_{}.wav".format(bare, i)
                if end - start < 100:
                    logging.info("SKIPPING short audio {}".format(target))
                else:
                    items.writerow(["{}_{}".format(bare, i), 0, end-start, triple[1], speaker, '_'.join([triple[0], triple[-1]])])
                    fragment.export(format='wav', out_f=target)
                    logging.info("Saved {}th trigram in {}".format(i, target))

    task = ABXpy.task.Task("data/flickr8k_abx.item", "phone", by="context", across="speaker")
    logging.info("Task statistics: {}".format(task.stats))
    logging.info("Generating triplets")
    if os.path.isfile("data/flickr8k_abx.triplets"):
        os.remove("data/flickr8k_abx.triplets")
    task.generate_triplets(output="data/flickr8k_abx.triplets")


def ed(x, y, normalized=None):
    return edit_distance(x, y)


def run_abx(feature_dir, triplet_file):
    root, _ = os.path.split(feature_dir)
    logging.info("Converting features {}".format(feature_dir))
    convert(feature_dir, root + "/features", load=_load_features_2019)
    logging.info("Computing distances")
    ABXpy.distances.distances.compute_distances(
            root + "/features",
            'features',
            triplet_file,
            root + "/distance",
            ed,
            normalized=True,
            n_cpu=16)
    logging.info("Computing scores")
    score.score(triplet_file, root + "/distance", root + "/score")
    analyze.analyze(triplet_file, root + "/score", root + "/analyze")
    data = pd.read_csv(root + "/analyze", delimiter='\t')
    return data


def compute_result(encoded_dir, triplets_fpath, output_dir):
    result = run_abx(encoded_dir, triplets_fpath)
    result.to_csv("{}/abx_flickr8k_analyze.csv".format(output_dir),
                  sep='\t', header=True, index=False)
    avg_error = _average("{}/abx_flickr8k_analyze.csv".format(output_dir),
                         "across")
    json.dump(dict(avg_abx_error=avg_error),
              open("{}/abx_flickr8k_result.json".format(output_dir), "w"))
    return avg_error


def abx(k=1000):
    from platalea.vq_encode import encode
    from vq_eval import experiments
    shutil.rmtree("data/flickr8k_abx_wav/", ignore_errors=True)
    prepare_abx(k=k)
    for modeldir in experiments("abx_flickr8k_result.json"):
    # for modeldir in ["experiments/vq-32-q1/"]:
    #for modeldir in ["experiments/vq-512-q1/"]:
        result = [json.loads(line) for line in open(modeldir + "result.json")]
        best = sorted(result, key=lambda x: x['recall']['10'], reverse=True)[0]['epoch']
        oldnet = torch.load("{}/net.{}.pt".format(modeldir, best))
        logging.info("Loading model from {} at epoch {}".format(modeldir, best))
        net = M.SpeechImage(oldnet.config)
        net.load_state_dict(oldnet.state_dict())
        net.cuda()
        encoded_dir = "{}/encoded/flickr8k_val/".format(modeldir)
        shutil.rmtree(encoded_dir, ignore_errors=True)
        Path(encoded_dir).mkdir(parents=True, exist_ok=True)
        logging.info("Encoding data")
        encode(net, "data/flickr8k_abx_wav/", encoded_dir)
        logging.info("Computing ABX")
        avg_error = compute_result(encoded_dir, "data/flickr8k_abx.triplets",
                                   modeldir)
        logging.info("Score: {}".format(avg_error))


def main():
    random.seed(123)
    logging.basicConfig(level=logging.INFO)
    abx(k=1000)


if __name__ == '__main__':
    main()
