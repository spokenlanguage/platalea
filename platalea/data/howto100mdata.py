import json
import pathlib

import torch
from sklearn.model_selection import train_test_split

from platalea.data.transcribeddataset import TranscribedDataset


class HowTo100MData(torch.utils.data.Dataset, TranscribedDataset):
    def __init__(self, root, feature_fname, id_map_fname, split='train',
                 downsampling_factor=None):
        self.root = root
        self.split = split
        self.feature_fname = feature_fname
        root_path = pathlib.Path(root)

        id_map_path = root_path / id_map_fname
        self.memmap_indices_by_id = _get_id_map(id_map_path, split, downsampling_factor)
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self.ids)

    def get_config(self):
        raise NotImplementedError()

    def evaluation(self):
        raise NotImplementedError()


def _get_id_map(id_map_path, split, downsampling_factor=None):
    id_map = _load_id_map(id_map_path)
    all_ids = list(id_map.keys())
    split_ids = _get_split(all_ids, split)
    down_sampled_ids = _get_down_sampled_ids(split_ids, downsampling_factor)
    return {id: id_map[id] for id in down_sampled_ids}


def _load_id_map(id_map_path):
    with open(id_map_path) as id_map_file:
        id_map = json.load(id_map_file)
    return id_map


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
