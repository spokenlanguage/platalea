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


def wordgrams(u, n):
    """Extract nonoverlapping segments of n words from utterance u."""
    xss = grams(u['words'], n)
    for xs in xss:
        s = dict(transcript=" ".join(x['word'] for x in xs),
                 words=list(xs),
                 audiopath=None)
        yield s


def shift_times(words):
    start = words[0]['start']
    startOffset = words[0]['startOffset']
    for word in words:
        word['start']       = word['start'] - start
        word['end']         = word['end'] - start
        word['startOffset'] = word['startOffset'] - startOffset
        word['endOffset']   = word['endOffset'] - startOffset
        yield word


def trigrams(xs):
    if len(xs) < 3:
        return []
    else:
        return [xs[0:3]] + trigrams(xs[1:])


def grams(xs, n):
    """Split sequence xs into non-overlapping segments of  length n."""
    if len(xs) < n:
        return []
    else:
        return [xs[:n]] + grams(xs[n:], n)


def fragments(xs):
    """Split sequence xs into non-overlapping segments of random lengths."""
    MIN = 3
    MAX = len(xs)-1
    if MAX-MIN <= MIN:
        return [xs]
    else:
        I = random.randint(MIN, MAX-MIN)
        return [xs[:I]] + fragments(xs[I:])


def deoov(xs):
    return [x for x in xs if not any(xi['phone'].startswith('oov') or xi['phone'].startswith('sil') for xi in x)]


def deoov_u(us):
    return [u for u in us if not any([p['phone'].startswith('oov') or p['phone'].startswith('sil') for w in u['words'] for p in w['phones']])]


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
    from prepare_flickr8k import load_alignment, good_alignment
    align = {key: value for key, value in load_alignment("data/datasets/flickr8k/fa.json").items() if good_alignment(value)}
    us = random.sample(list(align.values()), k)
    wav = Path("data/datasets/flickr8k/flickr_audio/wavs")
    out = Path("data/flickr8k_abx_wav")
    speakers = dict(line.split() for line in open("data/datasets/flickr8k/flickr_audio/wav2spk.txt"))
    out.mkdir(parents=True, exist_ok=True)
    with open("data/flickr8k_abx.item", "w") as itemout:
      with open("data/flickr8k_trigrams_fa.json", "w") as tri_fa:
        items = csv.writer(itemout, delimiter=' ', lineterminator='\n')
        items.writerow(["#file", "onset", "offset", "#phone", "speaker", "context", "lang"])
        for u in us:
            filename = os.path.split(u['audiopath'])[-1]
            speaker = speakers[filename]
            bare, _ = os.path.splitext(filename)
            grams = deoov(trigrams(phonemes(u)))
            logging.info("Loading audio from {}".format(filename))
            sound = pydub.AudioSegment.from_file(wav / filename)
            for i, gram in enumerate(grams):
                start = int(gram[0]['start']*1000)
                end = int(gram[-1]['end']*1000)
                triple = [phone['phone'].split('_')[0] for phone in gram]
                fragment = sound[start: end]
                target = out / "{}_{}.wav".format(bare, i)
                if end - start < 100:
                    logging.info("SKIPPING short audio {}".format(target))
                else:
                    items.writerow(["{}_{}".format(bare, i), 0, end-start, triple[1], speaker, '_'.join([triple[0], triple[-1]]), "en"])
                    fragment.export(format='wav', out_f=target)
                    # FIXME '_' should be '' for RSA?
                    word = '_'.join(phone['phone'].split('_')[0] for phone in gram)
                    tri_fa.write(json.dumps(dict(audiopath="{}".format(target),
                                                 transcript=word,
                                                 words=[dict(start=0,
                                                             end=sum([phone['duration'] for phone in gram]),
                                                             word=word,
                                                             alignedWord=word,
                                                             case='success',
                                                             phones=gram)])))
                    tri_fa.write("\n")
                    logging.info("Saved {}th trigram in {}".format(i, target))
    generate_triplets()
    generate_triplets(within_speaker=True)


def generate_triplets(within_speaker=False):
    if within_speaker:
        task = ABXpy.task.Task("data/flickr8k_abx.item", "phone",
                               by=["speaker", "context", "lang"])
        triplets = "data/flickr8k_abx_within.triplets"
    else:
        task = ABXpy.task.Task("data/flickr8k_abx.item", "phone",
                               by="context", across="speaker")
        triplets = "data/flickr8k_abx.triplets"
    logging.info("Task statistics: {}".format(task.stats))
    logging.info("Generating triplets")
    if os.path.isfile(triplets):
        os.remove(triplets)
    task.generate_triplets(output=triplets)


def prepare_fragments_fa(k=1000):
    logging.basicConfig(level=logging.INFO)
    from prepare_flickr8k import load_alignment, good_alignment
    align = {key: value for key, value in load_alignment("data/datasets/flickr8k/fa.json").items() if good_alignment(value)}
    if k is not None:
        us = random.sample(list(align.values()), k)
    else:
        us = list(align.values())
    wav = Path("data/datasets/flickr8k/flickr_audio/wavs")
    out = Path("data/flickr8k_fragments/")
    out.mkdir(parents=True, exist_ok=True)
    with open("data/flickr8k_fragments_fa.json", "w") as tri_fa:
        for u in us:
            filename = os.path.split(u['audiopath'])[-1]
            bare, _ = os.path.splitext(filename)
            grams = deoov(fragments(phonemes(u)))
            logging.info("Loading audio from {}".format(filename))
            sound = pydub.AudioSegment.from_file(wav / filename)
            for i, gram in enumerate(grams):
                start = int(gram[0]['start']*1000)
                end = int(gram[-1]['end']*1000)
                fragment = sound[start: end]
                target = out / "{}_{}.wav".format(bare, i)
                if end - start < 100:
                    logging.info("SKIPPING short audio {}".format(target))
                else:
                    fragment.export(format='wav', out_f=target)
                    tri_fa.write(json.dumps(dict(audiopath="{}".format(target),
                                                 transcript="<NA>",
                                                 words=[dict(start=0,
                                                             end=sum([phone['duration'] for phone in gram]),
                                                             word="<NA>",
                                                             alignedWord="<NA>",
                                                             case='success',
                                                             phones=gram)])))
                    tri_fa.write("\n")
                    logging.info("Saved {}th fragment in {}".format(i, target))


def prepare_wordgrams_fa(k=1000, n=1):
    logging.basicConfig(level=logging.INFO)
    from prepare_flickr8k import load_alignment, good_alignment
    align = {key: value for key, value in load_alignment("data/datasets/flickr8k/fa.json").items() if good_alignment(value)}
    if k is not None:
        us = random.sample(list(align.values()), k)
    else:
        us = list(align.values())
    wav = Path("data/datasets/flickr8k/flickr_audio/wavs")
    out = Path("data/flickr8k_wordgrams{}/".format(n))
    out.mkdir(parents=True, exist_ok=True)
    with open("data/flickr8k_wordgrams{}_fa.json".format(n), "w") as j:
        for u in us:
            filename = os.path.split(u['audiopath'])[-1]
            bare, _ = os.path.splitext(filename)
            grams = deoov_u(wordgrams(u, n))
            logging.info("Loading audio from {}".format(filename))
            sound = pydub.AudioSegment.from_file(wav / filename)
            for i, gram in enumerate(grams):
                start = int(gram['words'][0]['start']*1000)
                end = int(gram['words'][-1]['end']*1000)
                fragment = sound[start: end]
                target = out / "{}_{}.wav".format(bare, i)
                if end - start < 100:
                    logging.info("SKIPPING short audio {}".format(target))
                else:
                    fragment.export(format='wav', out_f=target)
                    gram['audiopath'] = "{}".format(target)
                    gram['words'] = list(shift_times(gram['words']))
                    j.write(json.dumps(gram))
                    j.write("\n")
                    logging.info("Saved {}th fragment in {}".format(i, target))


def prepare_abx_rep(directory, k=1000, within_speaker=False):
    import pickle
    import csv
    from prepare_flickr8k import make_indexer, load_alignment, good_alignment
    align = {key: value for key, value in load_alignment("data/datasets/flickr8k/fa.json").items() if good_alignment(value)}
    us = random.sample(list(align.values()), k)
    out = Path(directory) / "encoded/flickr8k_val_rep/"
    speakers = dict(line.split() for line in open("data/datasets/flickr8k/wav2spk.txt"))
    factors = json.load(open("{}/downsampling_factors.json".format(directory), "rb"))
    mode = 'trained'
    layer = 'codebook'
    global_inp = pickle.load(open("{}/global_input.pkl".format(directory), "rb"))
    global_act = pickle.load(open("{}/global_{}_{}.pkl".format(directory, mode, layer), "rb"))
    activation = dict(zip(global_inp['audio_id'], global_act[layer]))
    index = make_indexer(factors, layer)
    out.mkdir(parents=True, exist_ok=True)
    with open(Path(directory) / "flickr8k_abx_rep.item", "w") as itemout:
        items = csv.writer(itemout, delimiter=' ', lineterminator='\n')
        items.writerow(["#file", "onset", "offset", "#phone", "speaker", "context", "lang"])
        for u in us:
            filename = os.path.split(u['audiopath'])[-1]
            speaker = speakers[filename]
            bare, _ = os.path.splitext(filename)
            grams = deoov(trigrams(phonemes(u)))
            for i, gram in enumerate(grams):
                start = int(gram[0]['start']*1000)
                end = int(gram[-1]['end']*1000)
                triple = [phone['phone'].split('_')[0] for phone in gram]
                fragment = activation[filename][index(start): index(end)]
                target = out / "{}_{}.txt".format(bare, i)
                if end - start < 100 or fragment.shape[0] < 1:
                    logging.info("SKIPPING short audio {}".format(target))
                else:
                    items.writerow(["{}_{}".format(bare, i), 0, end-start, triple[1], speaker, '_'.join([triple[0], triple[-1]]), "en"])
                    np.savetxt(target, fragment.astype(int), fmt='%d')
                    logging.info("Saved activation of size {} for {}th trigram in {}".format(fragment.shape, i, target))
    logging.info("Saved item file in {}".format(Path(directory) / "flickr8k_abx_rep.item"))
    if within_speaker:

        task = ABXpy.task.Task("{}/flickr8k_abx_rep.item".format(directory), "phone", by=["speaker", "context", "lang"])
        triplets = "{}/flickr8k_abx_rep_within.triplets".format(directory)
    else:
        task = ABXpy.task.Task("{}/flickr8k_abx_rep.item".format(directory), "phone", by="context", across="speaker")
        triplets = "{}/flickr8k_abx_rep.triplets".format(directory)
    logging.info("Task statistics: {}".format(task.stats))
    logging.info("Generating triplets")
    if os.path.isfile(triplets):
        os.remove(triplets)
    task.generate_triplets(output=triplets)


def ed(x, y, normalized=None):
    return edit_distance(x, y)


def run_abx(feature_dir, triplet_file, distancefun=ed):
    logging.info("Running ABX on {} and {}".format(feature_dir, triplet_file))
    root, _ = os.path.split(feature_dir)
    logging.info("Converting features {}".format(feature_dir))
    convert(feature_dir, feature_dir + "/features", load=_load_features_2019)
    logging.info("Computing distances")
    ABXpy.distances.distances.compute_distances(
            feature_dir + "/features",
            'features',
            triplet_file,
            root + "/distance",
            distancefun,
            normalized=True,
            n_cpu=16)
    logging.info("Computing scores")
    score.score(triplet_file,  feature_dir + "/distance", feature_dir + "/score")
    analyze.analyze(triplet_file, feature_dir + "/score", feature_dir + "/analyze")
    data = pd.read_csv(feature_dir + "/analyze", delimiter='\t')
    return data


def compute_result(encoded_dir, triplets_fpath, output_dir,
                   within_speaker=False, distancefun=ed):
    result = run_abx(encoded_dir, triplets_fpath, distancefun)
    base, _ = os.path.splitext(os.path.split(triplets_fpath)[-1])
    if within_speaker:
        result.to_csv("{}/{}_analyze.csv".format(output_dir, base),
                      sep='\t', header=True, index=False)
        avg_error = _average("{}/{}_analyze.csv".format(output_dir, base),
                             "within")
        json.dump(dict(avg_abx_error=avg_error),
                  open("{}/{}_result.json".format(output_dir, base), "w"))
    else:
        result.to_csv("{}/{}_analyze.csv".format(output_dir, base),
                      sep='\t', header=True, index=False)
        avg_error = _average("{}/{}_analyze.csv".format(output_dir, base),
                             "across")
        json.dump(dict(avg_abx_error=avg_error),
                  open("{}/{}_result.json".format(output_dir, base), "w"))
    return avg_error


def abx(k=1000, within_speaker=False):
    from platalea.vq_encode import encode
    from vq_eval import experiments
    shutil.rmtree("data/flickr8k_abx_wav/", ignore_errors=True)
    prepare_abx(k=k, within_speaker=within_speaker)
    result = "abx_within_flickr8k_result.json" if within_speaker else "abx_flickr8k_result.json"
    for modeldir in experiments(result):
    #for modeldir in ["experiments/vq-32-q1/"]:
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
        triplets = ("{}/flickr8k_abx_rep_within.triplets" if within_speaker else "{}/flickr8k_abx_rep.triplets").format(modeldir)
        avg_error = compute_result(encoded_dir, triplets, modeldir, within_speaker=within_speaker)
        logging.info("Score: {}".format(avg_error))


def abx_rep(k=1000, within_speaker=False):
    from vq_eval import experiments
    result = "abx_rep_within_flickr8k_result.json" if within_speaker else "abx_rep_flickr8k_result.json"
    for modeldir in experiments(result):
        # FIXME this assumes the global_* files are already created
        encoded_dir = "{}/encoded/flickr8k_val_rep/".format(modeldir)
        shutil.rmtree(encoded_dir, ignore_errors=True)
        prepare_abx_rep(modeldir, k=k, within_speaker=within_speaker)
        Path(encoded_dir).mkdir(parents=True, exist_ok=True)
        logging.info("Computing ABX rep")
        triplets = "{}/flickr8k_abx_rep_within.triplets".format(modeldir) if within_speaker else "{}/flickr8k_abx_rep.triplets".format(modeldir)
        avg_error = compute_result(encoded_dir, triplets, modeldir, within_speaker=within_speaker)
        logging.info("Score: {}".format(avg_error))


def main():
    random.seed(123)
    within_speaker = True
    logging.basicConfig(level=logging.INFO)
    abx(k=1000, within_speaker=within_speaker)


if __name__ == '__main__':
    main()
