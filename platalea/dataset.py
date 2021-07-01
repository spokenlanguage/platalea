import torch
import torch.utils.data

from platalea.data.flickr8kdata import Flickr8KData
from platalea.data.howto100mdata import HowTo100MData
from platalea.data.librispeechdata import LibriSpeechData


def batch_audio(audios, max_frames=2048):
    """Merge audio captions. Truncate to max_frames. Pad with 0s."""
    mfcc_lengths = [len(cap[:max_frames, :]) for cap in audios]
    mfcc = torch.zeros(len(audios), max(mfcc_lengths), audios[0].size(1))
    for i, cap in enumerate(audios):
        end = mfcc_lengths[i]
        mfcc[i, :end] = cap[:end]
    return mfcc.permute(0, 2, 1), torch.tensor(mfcc_lengths)


def batch_text(texts):
    """Merge captions (from tuple of 1D tensor to 2D tensor). Pad with
    pad token."""
    char_lengths = [len(cap) for cap in texts]
    chars = torch.Tensor(len(texts), max(char_lengths)).long()
    chars.fill_(Flickr8KData.get_token_id(Flickr8KData.pad))
    for i, cap in enumerate(texts):
        end = char_lengths[i]
        chars[i, :end] = cap[:end]
    return chars, torch.tensor(char_lengths)


def batch_image(images):
    return torch.stack(images, 0)


def collate_fn(data, max_frames=2048):
    images, texts, audios = zip(* [(datum['image'],
                                    datum['text'],
                                    datum['audio']) for datum in data])
    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = batch_image(images)
    mfcc, mfcc_lengths = batch_audio(audios, max_frames=max_frames)
    chars, char_lengths = batch_text(texts)
    return dict(image=images, audio=mfcc, text=chars, audio_len=mfcc_lengths,
                text_len=char_lengths)


def collate_fn_speech(data, max_frames=2048):
    texts, audios = zip(* [(datum['text'],
                            datum['audio']) for datum in data])
    mfcc, mfcc_lengths = batch_audio(audios, max_frames=max_frames)
    chars, char_lengths = batch_text(texts)
    return dict(audio=mfcc, text=chars, audio_len=mfcc_lengths,
                text_len=char_lengths)


def howto100m_loader(root, video_features_subdir, id_map_fname,
                     batch_size=32, shuffle=False, split='train'):
    return torch.utils.data.DataLoader(dataset=HowTo100MData(root,
                                                             video_features_subdir,
                                                             id_map_fname,
                                                             split))


def flickr8k_loader(root, meta_fname, language, feature_fname,
                    split='train', batch_size=32, shuffle=False,
                    max_frames=2048,
                    downsampling_factor=None):
    return torch.utils.data.DataLoader(
        dataset=Flickr8KData(root=root,
                             feature_fname=feature_fname,
                             meta_fname=meta_fname,
                             split=split,
                             language=language,
                             downsampling_factor=downsampling_factor),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=lambda x: collate_fn(x, max_frames=max_frames))


def librispeech_loader(root, meta_fname, feature_fname,
                       split='train', batch_size=32, shuffle=False,
                       max_frames=2048,
                       downsampling_factor=None):
    return torch.utils.data.DataLoader(
        dataset=LibriSpeechData(root=root,
                                feature_fname=feature_fname,
                                meta_fname=meta_fname,
                                split=split,
                                downsampling_factor=downsampling_factor),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=lambda x: collate_fn_speech(x, max_frames=max_frames))
