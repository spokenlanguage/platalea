#!/usr/bin/env python3

"""
Preprocesses datasets
"""

import configargparse
import json
import logging
import numpy as np
import pathlib
import PIL.Image
import platalea.config
from soundfile import read
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


_device = platalea.config.device()


def preprocess(dataset_name):
    audio_feat_config = dict(type='mfcc', delta=True, alpha=0.97, n_filters=40,
                             window_size=0.025, frame_shift=0.010)
    images_feat_config = dict(model='resnet')
    if dataset_name == 'flickr8k':
        dataset_root = platalea.config.args.flickr8k_root
        audio_subdir = platalea.config.args.flickr8k_audio_subdir
        images_subdir = platalea.config.args.flickr8k_image_subdir
        dataset_path = pathlib.Path(dataset_root)
        flickr8k_audio_features(dataset_path, audio_subdir, audio_feat_config)
        flickr8k_image_features(dataset_path, images_subdir, images_feat_config)
    elif dataset_name == 'librispeech':
        dataset_root = platalea.config.args.librispeech_root
        dataset_path = pathlib.Path(dataset_root)
        librispeech_audio_features(dataset_path, audio_feat_config)
    else:
        raise NotImplementedError


def flickr8k_audio_features(dataset_path, audio_subdir, feat_config):
    directory = dataset_path / audio_subdir
    files = [line.split()[0] for line in open(dataset_path / 'flickr_audio' / 'wav2capt.txt')]
    paths = [directory / fn for fn in files]
    features = audio_features(paths, feat_config)
    torch.save(dict(features=features, filenames=files), dataset_path / 'mfcc_features.pt')


def flickr8k_image_features(dataset_path, images_subdir, feat_config):
    directory = dataset_path / images_subdir
    data = json.load(open(dataset_path / 'dataset.json'))
    files = [image['filename'] for image in data['images']]
    paths = [directory / fn for fn in files]
    features = torch.stack(image_features(paths, feat_config)).cpu()
    torch.save(dict(features=features, filenames=files), dataset_path / 'resnet_features.pt')


def librispeech_audio_features(dataset_path, feat_config):
    metadata = []
    paths = []
    set_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    for d1 in set_dirs:
        set_id = d1.name
        s = set_id.split('-')
        split = s[0]
        quality = s[1]
        reader_dirs = [d for d in d1.iterdir() if d.is_dir()]
        for d2 in reader_dirs:
            reader_id = d2.name
            chapter_dirs = [d for d in d2.iterdir() if d.is_dir()]
            for d3 in chapter_dirs:
                chapter_id = d3.name
                trn_path = d3 / '{}-{}.trans.txt'.format(reader_id, chapter_id)
                transcriptions = librispeech_load_trn(trn_path)
                for f in d3.glob('*.flac'):
                    fid = f.stem
                    sentid = fid.split('-')[2]
                    metadata.append(dict(
                        split=split, quality=quality, set_id=set_id,
                        spkrid=reader_id, chptid=chapter_id, sentid=sentid,
                        fileid=fid, fpath=str(f), trn=transcriptions[fid]))
                    paths.append(f)
    features = audio_features(paths, feat_config)
    # Saving features in memmap format
    memmap_fname = dataset_path / 'audio_features.memmap'
    start, end = save_audio_features_to_memmap(features, memmap_fname)
    for i, m in enumerate(metadata):
        m['audio_start'] = start[i]
        m['audio_end'] = end[i]
    with open(dataset_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f)


def save_audio_features_to_memmap(data, fname):
    num_lines = np.sum([d.shape[0] for d in data])
    fp = np.memmap(fname, dtype='float64', mode='w+', shape=(num_lines, 39))
    start = 0
    end = None
    S = []
    E = []
    for d in data:
        end = start + d.shape[0]
        fp[start:end, :] = d
        S.append(start)
        E.append(end)
        start = end
    return S, E


def librispeech_load_trn(path):
    with open(path) as f:
        lines = f.read().splitlines()
    transcriptions = {}
    for l in lines:
        s = l.split(maxsplit=1)
        transcriptions[s[0]] = s[1]
    return transcriptions


def image_features(paths, config):
    if config['model'] == 'resnet':
        model = models.resnet152(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
    elif config['model'] == 'vgg19':
        model = models.vgg19_bn(pretrained=True)
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    model.to(_device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    device = list(model.parameters())[0].device

    def one(path):
        logging.info("Extracting features from {}".format(path))
        im = PIL.Image.open(path)
        return prep_tencrop(im, model, device)

    return [one(path) for path in paths]


def prep_tencrop(im, model, device):
    # Adapted from: https://github.com/gchrupala/speech2image/blob/master/preprocessing/visual_features.py#L60

    # some functions such as taking the ten crop (four corners, center and
    # horizontal flip) normalise and resize.
    tencrop = transforms.TenCrop(224)
    tens = transforms.ToTensor()
    normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    resize = transforms.Resize(256, PIL.Image.ANTIALIAS)

    # there are some grayscale images in mscoco and places that the vgg and
    # resnet networks wont take
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im = tencrop(resize(im))
    im = torch.cat([normalise(tens(x)).unsqueeze(0) for x in im])
    im = im.to(device)
    activations = model(im)
    return activations.mean(0).squeeze()


def fix_wav(path):
    import wave
    logging.warning("Trying to fix {}".format(path))
    # fix wav file. In the flickr dataset there is one wav file with an
    # incorrect number of frames indicated in the header, causing it to be
    # unreadable by pythons wav read function. This opens the file with the
    # wave package, extracts the correct number of frames and saves a copy of
    # the file with a correct header

    file = wave.open(path, 'r')
    # derive the correct number of frames from the file
    frames = file.readframes(file.getnframes())
    # get all other header parameters
    params = file.getparams()
    file.close()
    # now save the file with a new header containing the correct number of
    # frames
    out_file = wave.open(path + '.fix', 'w')
    out_file.setparams(params)
    out_file.writeframes(frames)
    out_file.close()
    return path + '.fix'


def audio_features(paths, config):
    # Adapted from https://github.com/gchrupala/speech2image/blob/master/preprocessing/audio_features.py#L45
    from platalea.audio.features import get_fbanks, get_freqspectrum, get_mfcc, delta, raw_frames
    if config['type'] != 'mfcc' and config['type'] != 'fbank':
        raise NotImplementedError()
    output = []
    for cap in paths:
        logging.info("Processing {}".format(cap))
        try:
            data, fs = read(cap)
        except ValueError:
            # try to repair the file
            path = fix_wav(cap)
            data, fs = read(path)
        # get window and frameshift size in samples
        window_size = int(fs*config['window_size'])
        frame_shift = int(fs*config['frame_shift'])

        [frames, energy] = raw_frames(data, frame_shift, window_size)
        freq_spectrum = get_freqspectrum(frames, config['alpha'], fs,
                                         window_size)
        fbanks = get_fbanks(freq_spectrum, config['n_filters'], fs)
        if config['type'] == 'fbank':
            features = fbanks
        else:
            features = get_mfcc(fbanks)
            #  add the frame energy
            features = np.concatenate([energy[:, None], features], 1)

        # optionally add the deltas and double deltas
        if config['delta']:
            single_delta = delta(features, 2)
            double_delta = delta(single_delta, 2)
            features = np.concatenate([features, single_delta, double_delta], 1)
        output.append(torch.from_numpy(features))
    return output


if __name__ == '__main__':
    # Parsing command line
    doc = __doc__.strip("\n").split("\n", 1)
    parser = configargparse.get_argument_parser('platalea')
    parser.description = doc[0]
    parser.add_argument(
        'dataset_name', help='Name of the dataset to preprocess.',
        type=str, choices=['flickr8k', 'librispeech'])
    args, unknown_args = parser.parse_known_args()

    preprocess(args.dataset_name)
