import argparse
import json
import logging
import pickle
import torch

import platalea.dataset as D

torch.manual_seed(123)


def extract_trn(net, dataset, use_beam_decoding=False):
    d = dataset.evaluation()
    ref = d['text']
    with torch.no_grad():
        net.eval()
        if use_beam_decoding:
            hyp = net.transcribe(d['audio'], beam_size=10).tolist()
        else:
            hyp = net.transcribe(d['audio']).tolist()
    return hyp, ref


if __name__ == '__main__':
    batch_size = 16
    logging.basicConfig(level=logging.INFO)

    # Parse command line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('path', metavar='path', help='Model\'s path')
    parser.add_argument('-b', help='Use beam decoding', dest='use_beam_decoding',
                        action='store_true', default=False)
    args = parser.parse_args()

    # Loading config
    conf = pickle.load(open('config.pkl', 'rb'))

    logging.info('Loading data')
    data = dict(
        train=D.flickr8k_loader(split='train', batch_size=batch_size,
                                shuffle=False, feature_fname=conf['feature_fname'],
                                language=conf['language']),
        val=D.flickr8k_loader(split='val', batch_size=batch_size, shuffle=False,
                              feature_fname=conf['feature_fname'],
                              language=conf['language']))
    fd = D.Flickr8KData
    fd.le = conf['label_encoder']

    net = torch.load(args.path)




    trn = {}
    logging.info('Extracting transcriptions')
    for set_id, set_name in [('train', 'Training'), ('val', 'Validation')]:
        logging.info("{} set".format(set_name))
        d = data[set_id].dataset.evaluation()
        hyp, ref = extract_trn(net, data[set_id].dataset, args.use_beam_decoding)
        trn[set_id] = dict(hyp=hyp, ref=ref)
    json.dump(trn, open('trn.json', 'w'))
