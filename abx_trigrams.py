import pydub
import os.path
from pathlib import Path
import json
import platalea.ipa as ipa
import h5py
import logging

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
    
def chop(us):
    import csv
    wav = Path("data/datasets/flickr8k/flickr_audio/wavs")
    out = Path("data/flickr8k_abx_wav")
    with h5py.File("data/flickr8k_abx_top.features", "w") as features:
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
                #logging.info("3gram: {}".format(gram))
                start = int(gram[0]['start']*1000)
                end = int(gram[-1]['end']*1000)
                triple = [ ipa.arpa2ipa(phone['phone'].split('_')[0]) for phone in gram ]
                fragment = sound[start : end]
                target = out / bare / ("{}".format(i) + ".wav")
                items.writerow([Path(bare) / "{}".format(i), 0, end-start, triple[1], '_'.join(triple), '_'.join([triple[0], triple[-1]])])
                fragment.export(format='wav', out_f=target)
                logging.info("Saved {}th trigram in {}".format(i, target))


    

def main():
    logging.basicConfig(level=logging.INFO)
    align = {}
    for line in open("data/datasets/flickr8k/fa.json"):
        u = json.loads(line)
        u['audiopath'] =  os.path.split(u['audiopath'])[-1]
        align[u['audiopath']] = u
    chop(list(align.values())[:1000])


main()

