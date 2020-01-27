from config import CONFIG
import json
import os
from sklearn.preprocessing import LabelEncoder
import torch
import torch.utils.data


class Flickr8KData(torch.utils.data.Dataset):
    le = {}

    @classmethod
    def init_vocabularies(cls, dataset):
        # Set vocabulary for English
        le = LabelEncoder()
        tokens = ['<sos>', '<eos>', '<unk>', '<pad>'] + \
                 [c for d in dataset.split_data for c in d[2]]
        le.fit(tokens)
        cls.set_label_encoder(le, 'en')
        # Set vocabulary for Japanese
        le = LabelEncoder()
        tokens = ['<sos>', '<eos>', '<unk>', '<pad>'] + \
                 [c for d in dataset.split_data for c in d[3]]
        le.fit(tokens)
        cls.set_label_encoder(le, 'jp')

    @classmethod
    def set_label_encoder(cls, le, language='en'):
        cls.le[language] = le

    @classmethod
    def get_label_encoder(cls, language='en'):
        if language not in cls.le:
            raise ValueError('Vocabulary not initialized for {}'.format(language))
        return cls.le[language]

    @classmethod
    def get_token_id(cls, token, language='en'):
        return cls.get_label_encoder(language).transform([token])[0]

    @classmethod
    def vocabulary_size(cls, language=['en']):
        return len(cls.get_label_encoder(language).classes_)

    def __init__(self, root, feature_fname, split='train'):
        self.root = root
        self.split = split
        self.metadata = json.load(open(root + 'dataset_multilingual.json'))['images']
        # mapping from image id to list of caption id
        self.image_captions = {}
        for line in open(os.path.join(root + 'flickr_audio/wav2capt.txt')):
            audio_id, image_id, text_id = line.split()
            text_id = int(text_id[1:])
            self.image_captions[image_id] = self.image_captions.get(image_id, []) + [(text_id, audio_id)]

        # image, caption pairs
        self.split_data = []
        for image in self.metadata:
            if image['split'] == self.split:
                for text_id, audio_id in self.image_captions[image['filename']]:
                    self.split_data.append((
                        image['filename'],
                        audio_id,
                        image['sentences'][text_id]['raw'],
                        image['sentences'][text_id]['raw_jp']))

        # image and audio feature data
        image = torch.load(root + 'resnet_features.pt')
        self.image = dict(zip(image['filenames'], image['features']))
        audio = torch.load(root + feature_fname)
        self.audio = dict(zip(audio['filenames'], audio['features']))

    def caption2tensor(self, capt, language='en'):
        le = self.get_label_encoder(language)
        capt = [c if c in le.classes_ else '<unk>' for c in capt]
        capt = ['<sos>'] + capt + ['<eos>']
        return torch.Tensor(le.transform(capt))

    def __getitem__(self, index):
        sd = self.split_data[index]
        image = self.image[sd[0]]
        audio = self.audio[sd[1]]
        text = self.caption2tensor(sd[2], 'en')
        text_jp = self.caption2tensor(sd[3], 'jp')
        return dict(image_id=sd[0],
                    audio_id=sd[1],
                    image=image,
                    text=text,
                    text_jp=text_jp,
                    audio=audio,
                    gloss=sd[2])

    def __len__(self):
        return len(self.split_data)

    def evaluation(self):
        """Returns image features, caption features, and a boolean array
        specifying whether a caption goes with an image."""
        audio = []
        text  = []
        image = []
        matches = []
        for img in self.metadata:
            if img['split'] == self.split:
                image.append(self.image[img['filename']])
                for text_id, audio_id in self.image_captions[img['filename']]:
                    text.append(img['sentences'][text_id])
                    audio.append(self.audio[audio_id])
                    matches.append((len(audio)-1, len(image)-1))
        correct = torch.zeros(len(audio), len(image)).bool()
        for i, j in matches:
            correct[i, j] = True
        return dict(image=image, audio=audio, text=text, correct=correct)


def batch_audio(audios, max_frames=2048):
    """Merge audio captions. Truncate to max_frames. Pad with <pad> token."""
    mfcc_lengths = [len(cap[:max_frames, :]) for cap in audios]
    mfcc = torch.zeros(len(audios), max(mfcc_lengths), audios[0].size(1))
    for i, cap in enumerate(audios):
        end = mfcc_lengths[i]
        mfcc[i, :end] = cap[:end]
    return mfcc.permute(0, 2, 1), torch.tensor(mfcc_lengths)


def batch_text(texts, language='en'):
    """Merge captions (from tuple of 1D tensor to 2D tensor). Pad with
    <pad> token."""
    # FIXME this needs to be done properly, eventually
    char_lengths = [len(cap) for cap in texts]
    chars = torch.Tensor(len(texts), max(char_lengths)).long()
    chars.fill_(Flickr8KData.get_token_id('<pad>', language))
    for i, cap in enumerate(texts):
        end = char_lengths[i]
        chars[i, :end] = cap[:end]
    return chars, torch.tensor(char_lengths)


def batch_image(images):
    return torch.stack(images, 0)


def collate_fn(data, max_frames=2048):
    #data.sort(key=lambda x: len(x[1]), reverse=True)
    images, texts, texts_jp, audios = zip(* [(datum['image'],
                                              datum['text'],
                                              datum['text_jp'],
                                              datum['audio']) for datum in data])

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = batch_image(images)

    mfcc, mfcc_lengths = batch_audio(audios, max_frames=max_frames)
    chars, char_lengths = batch_text(texts, 'en')
    chars_jp, char_lengths_jp = batch_text(texts_jp, 'jp')

    return dict(image=images, audio=mfcc, text=chars, text_jp=chars_jp,
                audio_len=mfcc_lengths, text_len=char_lengths,
                text_len_jp=char_lengths_jp)


def flickr8k_loader(split='train', batch_size=32, shuffle=False,
                    max_frames=2048, feature_fname='mfcc_delta_features.pt'):
    return torch.utils.data.DataLoader(
        dataset=Flickr8KData(root=CONFIG['flickr8k_root'],
                             feature_fname=feature_fname,
                             split=split),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=lambda x: collate_fn(x, max_frames=max_frames))
