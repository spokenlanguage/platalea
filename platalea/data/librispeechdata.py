import json
import pathlib
import random

import numpy as np
import torch
import torch.utils

from platalea.data.transcribeddataset import TranscribedDataset


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