#!/usr/bin/env python3

"""
Preprocesses datasets
"""

import json
import logging
import os

import numpy as np
import pathlib
import PIL.Image
from moviepy.video.io.VideoFileClip import VideoFileClip

import platalea.hardware
import soundfile
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from platalea.experiments.config import get_argument_parser

args = get_argument_parser()
_audio_feat_config = dict(type='mfcc', delta=True, alpha=0.97, n_filters=40,
                          window_size=0.025, frame_shift=0.010)
_images_feat_config = dict(model='resnet')


def extract_audio_from_videos(video_dir_path, audio_dir_path):
    os.makedirs(audio_dir_path, exist_ok=True)
    for video_file_path in video_dir_path.iterdir():
        video = VideoFileClip(str(video_file_path))
        audio_file_name = video_file_path.name + '.wav'
        video.audio.write_audiofile(audio_dir_path / audio_file_name)
        video.close()


def preprocess_howto100m(dataset_path, audio_subdir, video_subdir, video_features_subdir):
    dataset_path_obj = pathlib.Path(dataset_path)
    audio_path = dataset_path_obj / audio_subdir
    video_path = dataset_path_obj / video_subdir
    if not audio_path.exists():
        extract_audio_from_videos(video_path, audio_path)

    extract_howto100m_audio_features(dataset_path_obj, audio_subdir, video_features_subdir, _audio_feat_config)


def find_howto100m_video_feature_files(ids, video_features_path):
    file_list = os.listdir(video_features_path)
    id_to_file_map = {file_name.split('.')[0]: file_name for file_name in file_list}

    missing_s3d_features = [id for id in ids if id not in id_to_file_map]
    if missing_s3d_features:
        raise FileNotFoundError(
            'Failed to find s3d features for the following ids: ' + missing_s3d_features)

    return [id_to_file_map[id] for id in ids]


def extract_howto100m_audio_features(dataset_path, audio_subdir, video_features_subdir, feat_config):
    audio_dir_path = dataset_path / audio_subdir
    file_names = os.listdir(audio_dir_path)
    paths = [audio_dir_path / fn for fn in file_names]
    features = audio_features(paths, feat_config)
    output_path = dataset_path / 'mfcc_features.memmap'
    starts, ends = save_audio_features_to_memmap(features, output_path)

    ids = [file_name.split('.')[0] for file_name in file_names]
    video_features_path = dataset_path / video_features_subdir
    video_features_files = find_howto100m_video_feature_files(ids, video_features_path)

    id_map = {id: {'audio_start': start, 'audio_end': end, 'video_feat_file': video}
              for id, start, end, video in zip(ids, starts, ends, video_features_files)}
    json.dump(id_map, open(dataset_path / 'id_map.json', 'w'))


def preprocess_flickr8k(dataset_path, audio_subdir, image_subdir):
    flickr8k_audio_features(pathlib.Path(dataset_path), audio_subdir, _audio_feat_config)
    flickr8k_image_features(pathlib.Path(dataset_path), image_subdir, _images_feat_config)


def preprocess_librispeech(dataset_path):
    librispeech_audio_features(pathlib.Path(dataset_path), _audio_feat_config)


def flickr8k_audio_features(dataset_path, audio_subdir, feat_config):
    directory = dataset_path / audio_subdir
    files = [line.split()[0] for line in open(dataset_path / 'wav2capt.txt')]
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


def save_audio_features_to_memmap(data, file_name):
    num_lines = np.sum([d.shape[0] for d in data])
    memmap = np.memmap(file_name, dtype='float64', mode='w+', shape=(num_lines, 39))
    start = 0
    start_indices = []
    end_indices = []
    for d in data:
        end = start + d.shape[0]
        memmap[start:end, :] = d
        start_indices.append(start)
        end_indices.append(end)
        start = end
    return start_indices, end_indices


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
    model.to(platalea.hardware.device())
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
            data, fs = soundfile.read(cap)
        except ValueError:
            # try to repair the file
            path = fix_wav(cap)
            data, fs = soundfile.read(path)
        # get window and frameshift size in samples
        window_size = int(fs * config['window_size'])
        frame_shift = int(fs * config['frame_shift'])

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
    args._parser.description = doc[0]
    args.add_argument(
        'dataset_name', help='Name of the dataset to preprocess.',
        type=str, choices=['flickr8k', 'librispeech', 'howto100m-encc'])
    args.enable_help()
    args.parse()

    if args.dataset_name == "flickr8k":
        preprocess_flickr8k(args.flickr8k_root, args.flickr8k_audio_subdir, args.flickr8k_image_subdir)
    elif args.dataset_name == "librispeech":
        preprocess_librispeech(args.librispeech_root)
    if args.dataset_name == "howto100m-encc":
        preprocess_howto100m(args.howto100m_root, args.howto100m_audio_subdir, args.howto100m_video_subdir)
