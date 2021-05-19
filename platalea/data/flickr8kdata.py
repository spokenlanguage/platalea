import json
import pathlib
import pickle
import random

import torch
import torch.utils

from platalea.data.transcribeddataset import TranscribedDataset


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