import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import PIL.Image
import json
import logging
from scipy.io.wavfile import read
import numpy
import platalea.config
import pathlib


_device = platalea.config.device()


def flickr8k_features(dataset_path=platalea.config.args.data_root,
                      audio_subdir=platalea.config.args.audio_subdir,
                      images_subdir=platalea.config.args.image_subdir):
    audio_config = dict(dataset_path=pathlib.Path(dataset_path), audio_subdir=audio_subdir, type='mfcc',
                        delta=True, alpha=0.97, n_filters=40, window_size=0.025,
                        frame_shift=0.010)
    images_config = dict(dataset_path=pathlib.Path(dataset_path), images_subdir=images_subdir, model='resnet')
    flickr8k_audio_features(audio_config)
    flickr8k_image_features(images_config)


def flickr8k_audio_features(config):
    directory = config['dataset_path'] / config['audio_subdir']
    files = [line.split()[0] for line in open(config['dataset_path'] / 'wav2capt.txt')]
    paths = [directory / fn for fn in files]
    features = audio_features(paths, config)
    torch.save(dict(features=features, filenames=files), config['dataset_path'] / 'mfcc_features.pt')


def flickr8k_image_features(config):
    directory = config['dataset_path'] / config['images_subdir']
    data = json.load(open(config['dataset_path'] / 'dataset.json'))
    files = [image['filename'] for image in data['images']]
    paths = [directory / fn for fn in files]

    features = image_features(paths, config).cpu()
    torch.save(dict(features=features, filenames=files), config['dataset_path'] / 'resnet_features.pt')


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

    return torch.stack([one(path) for path in paths])


def prep_tencrop(im, model, device):
    # Adapted from: https://github.com/gchrupala/speech2image/blob/master/preprocessing/visual_features.py#L60

    # some functions such as taking the ten crop (four corners, center and
    # horizontal flip) normalise and resize.
    tencrop = transforms.TenCrop(224)
    tens = transforms.ToTensor()
    normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    resize = transforms.Resize(256, PIL.Image.ANTIALIAS)

    im = tencrop(resize(im))
    im = torch.cat([normalise(tens(x)).unsqueeze(0) for x in im])
    im = im.to(device)
    # there are some grayscale images in mscoco that the vgg and resnet
    # networks wont take
    if not im.size()[1] == 3:
        im = im.expand(im.size()[0], 3, im.size()[2], im.size()[3])
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
            input_data = read(cap)
        except ValueError:
            # try to repair the file
            path = fix_wav(cap)
            input_data = read(path)
        # sampling frequency
        fs = input_data[0]
        # get window and frameshift size in samples
        window_size = int(fs*config['window_size'])
        frame_shift = int(fs*config['frame_shift'])

        [frames, energy] = raw_frames(input_data, frame_shift, window_size)
        freq_spectrum = get_freqspectrum(frames, config['alpha'], fs,
                                         window_size)
        fbanks = get_fbanks(freq_spectrum, config['n_filters'], fs)
        if config['type'] == 'fbank':
            features = fbanks
        else:
            features = get_mfcc(fbanks)
            #  add the frame energy
            features = numpy.concatenate([energy[:, None], features], 1)

        # optionally add the deltas and double deltas
        if config['delta']:
            single_delta = delta(features, 2)
            double_delta = delta(single_delta, 2)
            features = numpy.concatenate([features, single_delta, double_delta], 1)
        output.append(torch.tensor(features))
    return output
