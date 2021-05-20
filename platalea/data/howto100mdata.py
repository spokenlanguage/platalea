import pathlib

import torch
from sklearn.model_selection import train_test_split

from platalea.data.transcribeddataset import TranscribedDataset


class HowTo100MData(torch.utils.data.Dataset, TranscribedDataset):
    def __init__(self, root, feature_fname, id_fname, split='train',
                 downsampling_factor=None):
        self.root = root
        self.split = split
        self.feature_fname = feature_fname
        root_path = pathlib.Path(root)

        id_file_path = root_path / id_fname
        ids = _get_ids(id_file_path, downsampling_factor)
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def get_config(self):
        raise NotImplementedError()

    def evaluation(self):
        raise NotImplementedError()


def _get_ids(id_file_path, split, downsampling_factor=None):
    ids = _read_ids_from_file(id_file_path)
    ids = _get_split(ids, split)
    return _get_down_sampled_ids(ids, downsampling_factor)


def _read_ids_from_file(id_file_path):
    with open(id_file_path) as id_file:
        ids = id_file.readlines()
    return ids


def _get_split(ids, split):
    train_ids, test_ids = train_test_split(ids, test_size=1000, random_state=0)

    if split == 'train':
        return train_ids
    if split == 'test':
        return test_ids
    raise ValueError('split should be either "train" or "test". '
                     'Instead encountered illegal value: {}'
                     .format(split))


def _get_down_sampled_ids(all_ids, downsampling_factor):
    if downsampling_factor is None:
        num_examples = len(all_ids)
    else:
        num_examples = len(all_ids) // downsampling_factor
    return all_ids[:num_examples]
