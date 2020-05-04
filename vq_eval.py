import torch
import platalea.basicvq as M
import json
import glob
from platalea.vq_encode import evaluate_zerospeech
import pathlib
import logging
logging.basicConfig(level=logging.INFO)

for modeldir in glob.glob("experiments/vq-*/"):
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
    
