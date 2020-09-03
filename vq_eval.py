import argparse
import torch
import platalea.basicvq as M
import json
import glob
from platalea.vq_encode import evaluate_zerospeech
import pathlib
from pathlib import Path
import logging
import os.path
import numpy as np
import pickle
import lyz.methods

logging.basicConfig(level=logging.INFO)


def experiments(outfile=None):
    """Return the list of directories which do not contain the result file
named `outfile`."""
    dirs = glob.glob("experiments/vq-*/")
    if outfile is None:
        return dirs
    else:
        out = []
        for d in dirs:
            path = Path(d) / outfile
            if os.path.isfile(path):
                logging.info("File {} exists, skipping directory".format(path))
            else:
                out.append(d)
        return out


def zerospeech_baseline():
    for modeldir in experiments("vq_base_result.json"):
        result = [ json.loads(line) for line in open(modeldir + "/result.json") ]
        best = result[0]['epoch']
        oldnet = torch.load("{}/net.{}.pt".format(modeldir, best))
        logging.info("Loading model from {} at epoch {}".format(modeldir, best))
        net = M.SpeechImage(oldnet.config)
        net.load_state_dict(oldnet.state_dict())
        net.cuda()
        encoded_dir = "{}/encoded-base/2019/english/test/".format(modeldir)
        pathlib.Path(encoded_dir).mkdir(parents=True, exist_ok=True)
        logging.info("Encoding and evaluating zerospeech data")
        scores = evaluate_zerospeech(net, outdir=encoded_dir)
        logging.info("Result: {}".format(scores['2019']['english']['scores']))
        scores['epoch' ]=  best
        scores['modelpath'] =  "{}/net.{}.pt".format(modeldir, best)
        json.dump(scores, open("{}/vq_base_result.json".format(modeldir), "w"))


def zerospeech():
    for modeldir in experiments("vq_result.json"):
        result = [ json.loads(line) for line in open(modeldir + "result.json") ]
        best = sorted(result, key=lambda x: x['recall']['10'], reverse=True)[0]['epoch']
        oldnet = torch.load("{}/net.{}.pt".format(modeldir, best))
        logging.info("Loading model from {} at epoch {}".format(modeldir, best))
        net = M.SpeechImage(oldnet.config)
        net.load_state_dict(oldnet.state_dict())
        net.cuda()
        encoded_dir = "{}/encoded/2019/english/test/".format(modeldir)
        pathlib.Path(encoded_dir).mkdir(parents=True, exist_ok=True)
        logging.info("Encoding and evaluating zerospeech data")
        scores = evaluate_zerospeech(net, outdir=encoded_dir)
        logging.info("Result: {}".format(scores['2019']['english']['scores']))
        scores['epoch' ]=  best
        scores['modelpath'] =  "{}/net.{}.pt".format(modeldir, best)
        json.dump(scores, open("{}/vq_result.json".format(modeldir), "w"))


def prepare_rsa():
    from prepare_flickr8k import save_data, make_factors
    for modeldir in experiments("ed_rsa.json"):
        result = [ json.loads(line) for line in open(modeldir + "result.json") ]
        best = sorted(result, key=lambda x: x['recall']['10'], reverse=True)[0]['epoch']
        oldnet = torch.load("{}/net.{}.pt".format(modeldir, best))
        logging.info("Loading model from {} at epoch {}".format(modeldir, best))
        net = M.SpeechImage(oldnet.config)
        net.load_state_dict(oldnet.state_dict())
        net.cuda()
        json.dump(make_factors(net), open("{}/downsampling_factors.json".format(modeldir), "w"))
        net = M.SpeechImage(oldnet.config)
        net_rand = M.SpeechImage(oldnet.config)
        net.load_state_dict(oldnet.state_dict())
        net.cuda()
        net_rand.cuda()
        save_data([('trained', net), ('random', net_rand)], modeldir, batch_size=8)


def prepare_rsa_trigrams():
    from prepare_flickr8k import save_data_trigrams, make_factors
    for modeldir in experiments("ed_rsa_trigrams.json"):
        result = [ json.loads(line) for line in open(modeldir + "/result.json") ]
        best = sorted(result, key=lambda x: x['recall']['10'], reverse=True)[0]['epoch']
        oldnet = torch.load("{}/net.{}.pt".format(modeldir, best))
        logging.info("Loading model from {} at epoch {}".format(modeldir, best))
        net = M.SpeechImage(oldnet.config)
        net.load_state_dict(oldnet.state_dict())
        net.cuda()
        json.dump(make_factors(net), open("{}/downsampling_factors.json".format(modeldir), "w"))
        net = M.SpeechImage(oldnet.config)
        net_rand = M.SpeechImage(oldnet.config)
        net.load_state_dict(oldnet.state_dict())
        net.cuda()
        net_rand.cuda()
        save_data_trigrams([('trained', net), ('random', net_rand)], "{}/trigrams".format(modeldir), batch_size=8)

def prepare_rsa_fragments():
    from prepare_flickr8k import save_data_fragments, make_factors
    for modeldir in experiments("ed_rsa_fragments.json"):
        result = [ json.loads(line) for line in open(modeldir + "/result.json") ]
        best = sorted(result, key=lambda x: x['recall']['10'], reverse=True)[0]['epoch']
        oldnet = torch.load("{}/net.{}.pt".format(modeldir, best))
        logging.info("Loading model from {} at epoch {}".format(modeldir, best))
        net = M.SpeechImage(oldnet.config)
        net.load_state_dict(oldnet.state_dict())
        net.cuda()
        json.dump(make_factors(net), open("{}/downsampling_factors.json".format(modeldir), "w"))
        net = M.SpeechImage(oldnet.config)
        net_rand = M.SpeechImage(oldnet.config)
        net.load_state_dict(oldnet.state_dict())
        net.cuda()
        net_rand.cuda()
        save_data_fragments([('trained', net), ('random', net_rand)], "{}/fragments".format(modeldir), batch_size=8)

def prepare_rsa_wordgrams(n=1):
    from prepare_flickr8k import save_data_wordgrams, make_factors
    for modeldir in experiments("ed_rsa_wordgrams{}.json".format(n)):
        result = [ json.loads(line) for line in open(modeldir + "/result.json") ]
        best = sorted(result, key=lambda x: x['recall']['10'], reverse=True)[0]['epoch']
        oldnet = torch.load("{}/net.{}.pt".format(modeldir, best))
        logging.info("Loading model from {} at epoch {}".format(modeldir, best))
        net = M.SpeechImage(oldnet.config)
        net.load_state_dict(oldnet.state_dict())
        net.cuda()
        json.dump(make_factors(net), open("{}/downsampling_factors.json".format(modeldir), "w"))
        net = M.SpeechImage(oldnet.config)
        net_rand = M.SpeechImage(oldnet.config)
        net.load_state_dict(oldnet.state_dict())
        net.cuda()
        net_rand.cuda()
        batch_size = 32 if n > 2 else 128 
        save_data_wordgrams([('trained', net), ('random', net_rand)], "{}/wordgrams{}".format(modeldir, n), n, batch_size=batch_size)


def rsa(modeldirs):
    from lyz.methods import ed_rsa
    terciles = (0.0, 1/3, 2/3, 1.0)
    for mdldir in modeldirs:
        logging.info("Processing {}".format(mdldir))
        cor = ed_rsa(mdldir, layers=['codebook'], test_size=1/2,
                     quantiles=terciles)
        logging.info("RSA for {}: {}".format(mdldir, json.dumps(cor, indent=2)))
        json.dump(cor, open("{}/ed_rsa.json".format(mdldir), "w"))


def rsa_trigrams(modeldirs):
    from lyz.methods import ed_rsa
    for modeldir in modeldirs:
        logging.info("Processing {}".format(modeldir))
        cor = ed_rsa("{}/trigrams".format(modeldir), layers=['codebook'], test_size=1/2)
        logging.info("RSA on trigrams for {}: {}".format(modeldir, json.dumps(cor, indent=2)))
        json.dump(cor, open("{}/ed_rsa_trigrams.json".format(modeldir), "w"))
        
def rsa_fragments(quantiles=lyz.methods.DECILES):
    from lyz.methods import ed_rsa
    for modeldir in experiments("ed_rsa_fragments.json"):
        logging.info("Processing {}".format(modeldir))
        cor = ed_rsa("{}/fragments".format(modeldir), layers=['codebook'], test_size=1/2, quantiles=quantiles)
        logging.info("RSA on fragments for {}: {}".format(modeldir, json.dumps(cor, indent=2)))
        json.dump(cor, open("{}/ed_rsa_fragments.json".format(modeldir), "w"))

def rsa_wordgrams(n=1):
    from lyz.methods import ed_rsa
    for modeldir in experiments("ed_rsa_wordgrams{}.json".format(n)):
        logging.info("Processing {}".format(modeldir))
        cor = ed_rsa("{}/wordgrams{}".format(modeldir, n), layers=['codebook'], test_size=1/2)
        logging.info("RSA on fragments for {}: {}".format(modeldir, json.dumps(cor, indent=2)))
        json.dump(cor, open("{}/ed_rsa_wordgrams{}.json".format(modeldir, n), "w"))


def prepare_and_run_rsa_wordgrams():
    from abx_trigrams import prepare_wordgrams_fa
    for n in [1, 5]:
        prepare_wordgrams_fa(k=None, n=n)
        prepare_rsa_wordgrams(n=n)
        rsa_wordgrams(n=n)
        
def local_diag(modeldirs):
    from lyz.methods import local_diagnostic
    for mdldir in modeldirs:
        logging.info("Running local diagnostic on {}".format(mdldir))
        output_dir = Path(mdldir) / 'local'
        output_dir.mkdir(parents=True, exist_ok=True)

        logging.info("Local diagnostic")
        config = dict(directory=mdldir,
                      output=output_dir,
                      hidden=None,
                      epochs=40,
                      layers=['codebook'],
                      runs=1)
        local_diagnostic(config)


def rle_compression_ratio():
    def rle(seq):
        "Compute run-length encoding of `seq`."
        from itertools import groupby
        return [ (k, len(list(v))) for k, v in groupby(seq) ]

    def rle_string(seq):
        return ''.join([ k if v == 1 else "{}{}".format(k,v) for k,v in rle(seq) ])

    def rle_ratio(seq):
        return len(seq) / len(rle_string(seq))

    def directory_ratio(directory, layer='codebook', mode='trained'):
        ratios = []
        data = pickle.load(open("{}/global_{}_{}.pkl".format(directory, mode, layer), "rb"))
        act = data[layer]
        ratio = np.mean([ rle_ratio([ chr(x) for x in item.argmax(axis=1)]) for item in act ])
        return ratio

    for modeldir in experiments("rle_compression.json"):
        logging.info("Processing {}".format(modeldir))
        ratio = directory_ratio(modeldir, layer='codebook', mode='trained')
        json.dump(dict(ratio=ratio), open("{}/rle_compression.json".format(modeldir), "w"))


if __name__ == '__main__':
    # Parsing command line
    doc = __doc__.strip("\n").split("\n", 1)
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-m', '--method', help='Method for analysis',
                        choices=['rsa', 'rsa_trigrams', 'dc'])
    args = parser.parse_args()

    if args.method == 'rsa':
        rsa(experiments("ed_rsa.json"))
    elif args.method == 'rsa_trigrams':
        rsa_trigrams(experiments("ed_rsa_trigrams.json"))
    elif args.method == 'dc':
        local_diag(experiments("local/local_diagnostic.json"))
