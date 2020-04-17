import configargparse
import logging
import pickle
import os
from shutil import copyfile
import torch

import platalea.asr as M1
import platalea.dataset as D
import platalea.text_image as M2
from utils.copy_best import copy_best
from utils.extract_transcriptions import extract_trn

torch.manual_seed(123)


batch_size = 8

logging.basicConfig(level=logging.INFO)

# Parse command line parameters
parser = configargparse.get_argument_parser('platalea')
parser.add_argument(
    '--asr_model_dir',
    help='Path to the directory where the pretrained ASR model is stored',
    dest='asr_model_dir', type=str, action='store')
args, unknown_args = parser.parse_known_args()

factors = [3, 9, 27, 81, 243]
lz = len(str(abs(factors[-1])))
for ds_factor in factors:
    logging.info('Loading data')
    data = dict(
        train=D.flickr8k_loader(split='train', batch_size=batch_size,
                                shuffle=True, downsampling_factor=ds_factor),
        val=D.flickr8k_loader(split='val', batch_size=batch_size,
                              shuffle=False))
    fd = D.Flickr8KData
    if args.asr_model_dir:
        config_fpath = os.path.join(args.asr_model_dir, 'config.pkl')
        config = pickle.load(open(config_fpath, 'rb'))
        fd.le = config['label_encoder']
    else:
        fd.init_vocabulary(data['train'].dataset)
        # Saving config
        pickle.dump(data['train'].dataset.get_config(),
                    open('config.pkl', 'wb'))

    if args.asr_model_dir:
        net = torch.load(os.path.join(args.asr_model_dir, 'net.best.pt'))
    else:
        logging.info('Building ASR model')
        config = M1.get_default_config()
        net = M1.SpeechTranscriber(config)
        run_config = dict(max_norm=2.0, max_lr=2 * 1e-4, epochs=32, opt='adam')
        logging.info('Training ASR')
        M1.experiment(net, data, run_config)
        suffix = str(ds_factor).zfill(lz)
        res_fname = 'result_asr_{}.json'.format(suffix)
        copyfile('result.json', res_fname)
        net_fname = 'asr_{}.best.pt'.format(ds_factor)
        copy_best(res_fname, net_fname, experiment_type='asr')
        net = torch.load(net_fname)

    logging.info('Extracting ASR transcriptions')
    for set_name in ['train', 'val']:
        ds = data[set_name].dataset
        hyp_asr, ref_asr = extract_trn(net, ds, use_beam_decoding=True)
        # Replacing original transcriptions with ASR's output
        for i in range(len(hyp_asr)):
            item = ds.split_data[i]
            if item[2] == ref_asr[i]:
                ds.split_data[i] = (item[0], item[1], hyp_asr[i])
            else:
                msg = 'Extracted reference #{} ({}) doesn\'t match dataset\'s \
                        one ({}) for {} set.'
                msg = msg.format(i, ref_asr[i], ds.split_data[i][3], set_name)
                logging.warning(msg)

    if args.asr_model_dir:
        # Saving config for text-image model
        pickle.dump(data['train'].dataset.get_config(),
                    open('config.pkl', 'wb'))

    logging.info('Building model text-image')
    net = M2.TextImage(M2.get_default_config())
    run_config = dict(max_lr=2 * 1e-4, epochs=32)

    logging.info('Training text-image')
    M2.experiment(net, data, run_config)
    suffix = str(ds_factor).zfill(lz)
    res_fname = 'result_text_image_{}.json'.format(suffix)
    copyfile('result.json', res_fname)
    net_fname = 'ti_{}.best.pt'.format(ds_factor)
    copy_best(res_fname, net_fname)
