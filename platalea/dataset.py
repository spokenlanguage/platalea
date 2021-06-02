import json
import numpy as np
import pathlib
import pickle
import random
from sklearn.preprocessing import LabelEncoder
import torch
import torch.utils.data


class TranscribedDataset():
    le = None
    sos = '<sos>'
    eos = '<eos>'
    pad = '<pad>'
    unk = '<unk>'

    @classmethod
    def init_vocabulary(cls, transcriptions):
        cls.le = LabelEncoder()
        tokens = [cls.sos, cls.eos, cls.unk, cls.pad] + \
                 [c for t in transcriptions for c in t]
        cls.le.fit(tokens)

    @classmethod
    def get_label_encoder(cls):
        if cls.le is None:
            raise ValueError('Vocabulary not initialized.')
        return cls.le

    @classmethod
    def get_token_id(cls, token):
        return cls.get_label_encoder().transform([token])[0]

    @classmethod
    def vocabulary_size(cls):
        return len(cls.get_label_encoder().classes_)

    @classmethod
    def caption2tensor(cls, capt):
        le = cls.get_label_encoder()
        capt = [c if c in le.classes_ else cls.unk for c in capt]
        capt = [cls.sos] + capt + [cls.eos]
        return torch.Tensor(le.transform(capt))


class Flickr8KData(torch.utils.data.Dataset, TranscribedDataset):
    @classmethod
    def init_vocabulary(cls, dataset):
        transcriptions = [sd[2] for sd in dataset.split_data]
        TranscribedDataset.init_vocabulary(transcriptions)

    def __init__(self, root, feature_fname, meta_fname, split='train', language='en',
                 downsampling_factor=None):
        self.root = root
        self.split = split
        self.feature_fname = feature_fname
        self.language = language
        if language == 'en':
            self.text_key = 'raw'
        elif language == 'jp':
            self.text_key = 'raw_jp'
        else:
            raise ValueError('Language {} not supported.'.format(language))
        self.root = root
        self.split = split
        self.language = language
        root_path = pathlib.Path(root)
        # Loading label encoder
        module_path = pathlib.Path(__file__).parent
        with open(module_path / 'label_encoders.pkl', 'rb') as f:
            self.__class__.le = pickle.load(f)[language]
        # Loading metadata
        with open(root_path / meta_fname) as fmeta:
            metadata = json.load(fmeta)['images']
        # Loading mapping from image id to list of caption id
        self.image_captions = {}
        with open(root_path / 'flickr_audio' / 'wav2capt.txt') as fwav2capt:
            for line in fwav2capt:
                audio_id, image_id, text_id = line.split()
                text_id = int(text_id[1:])
                self.image_captions[image_id] = self.image_captions.get(image_id, []) + [(text_id, audio_id)]
        # Creating image, caption pairs
        self.split_data = []
        for image in metadata:
            if image['split'] == self.split:
                fname = image['filename']
                for text_id, audio_id in self.image_captions[fname]:
                    # In the reduced dataset containing only sentences with
                    # translations, removed sentences are replaced by 'None' to
                    # keep the index of the sentence fixed, so that we can
                    # still retrieve them based on text_id.
                    # TODO: find a nicer way to handle this
                    if image['sentences'][text_id] is not None:
                        if self.text_key in image['sentences'][text_id]:
                            self.split_data.append((
                                fname,
                                audio_id,
                                image['sentences'][text_id][self.text_key]))

        # Downsampling
        if downsampling_factor is not None:
            num_examples = int(len(self.split_data) // downsampling_factor)
            self.split_data = random.sample(self.split_data, num_examples)

        # image and audio feature data
        image = torch.load(root_path / 'resnet_features.pt')
        self.image = dict(zip(image['filenames'], image['features']))
        audio = torch.load(root_path / feature_fname)
        self.audio = dict(zip(audio['filenames'], audio['features']))

    def __getitem__(self, index):
        sd = self.split_data[index]
        image = self.image[sd[0]]
        audio = self.audio[sd[1]]
        text = self.caption2tensor(sd[2])
        return dict(image_id=sd[0],
                    audio_id=sd[1],
                    image=image,
                    text=text,
                    audio=audio,
                    gloss=sd[2])

    def __len__(self):
        return len(self.split_data)

    def get_config(self):
        return dict(feature_fname=self.feature_fname,
                    label_encoder=self.get_label_encoder(),
                    language=self.language)

    def evaluation(self):
        """Returns image features, audio features, caption features, and a
        boolean array specifying whether a caption goes with an image."""
        audio = []
        text = []
        image = []
        matches = []
        image2idx = {}
        for sd in self.split_data:
            # Add image
            if sd[0] in image2idx:
                image_idx = image2idx[sd[0]]
            else:
                image_idx = len(image)
                image2idx[sd[0]] = image_idx
                image.append(self.image[sd[0]])
            # Add audio and text
            audio.append(self.audio[sd[1]])
            text.append(sd[2])
            matches.append((len(audio) - 1, image_idx))
        correct = torch.zeros(len(audio), len(image)).bool()
        for i, j in matches:
            correct[i, j] = True
        return dict(image=image, audio=audio, text=text, correct=correct)

    def is_slt(self):
        return self.language != 'en'

    def split_sentences(self, sentences):
        if self.language == 'jp':
            return sentences
        else:
            return [s.split() for s in sentences]


class LibriSpeechData(torch.utils.data.Dataset, TranscribedDataset):
    @classmethod
    def init_vocabulary(cls, dataset):
        transcriptions = [m['trn'] for m in dataset.metadata]
        TranscribedDataset.init_vocabulary(transcriptions)

    def __init__(self, root, feature_fname, meta_fname, split='train',
                 downsampling_factor=None):
        # 'val' set in flickr8k corresponds to 'dev' in librispeech
        if split == 'val':
            split = 'dev'
        self.root = root
        self.split = split
        self.feature_fname = feature_fname
        root_path = pathlib.Path(root)
        with open(root_path / meta_fname) as fmeta:
            self.metadata = json.load(fmeta)
            self.num_lines = self.metadata[-1]['audio_end']
        if downsampling_factor is not None:
            num_examples = len(self.metadata) // downsampling_factor
            self.metadata = random.sample(self.metadata, num_examples)
        # filter examples based on split
        meta = []
        for ex in self.metadata:
            if ex['split'] == self.split:
                meta.append(ex)
        self.metadata = meta
        # load audio features
        self.audio = np.memmap(root_path / feature_fname, dtype='float64',
                               mode='r', shape=(self.num_lines, 39))

    def __getitem__(self, index):
        sd = self.metadata[index]
        audio = torch.from_numpy(self.audio[sd['audio_start']:sd['audio_end']])
        text = self.caption2tensor(sd['trn'])
        return dict(audio_id=sd['fileid'], text=text, audio=audio)

    def __len__(self):
        return len(self.metadata)

    def get_config(self):
        return dict(feature_fname=self.feature_fname,
                    label_encoder=self.get_label_encoder())

    def evaluation(self):
        """Returns audio features with corresponding caption"""
        audio = []
        text = []
        for ex in self.metadata:
            text.append(ex['trn'])
            a = torch.from_numpy(self.audio[ex['audio_start']:ex['audio_end']])
            audio.append(a)
        return dict(audio=audio, text=text)


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
