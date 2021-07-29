import shutil
from pathlib import Path
from unittest import TestCase

import numpy as np

from platalea.dataset import _get_id_map, HowTo100MData


class Howto100mProcessingCase(TestCase):
    _dataset_path = Path('./testdata/howto100msample')  # Should only be accessed by setup.

    test_dataset_path = Path('./testdata/temp_howto100msample')
    id_map_file_name = 'long_id_map.json'
    audio_subdir = 'audio'
    video_subdir = 'video'
    video_features_subdir = 's3d_features'
    audio_features_file_name = 'dummy_mfcc_features.memmap'

    def test_get_id_map_number_of_results(self):
        id_map_path = self.test_dataset_path / self.id_map_file_name
        assert len(_get_id_map(id_map_path, 'test')) == 1000, \
            'By default 1000 ids should go to the test set'

        assert len(_get_id_map(id_map_path, 'train')) == 3422, \
            'The other 3422 ids in the test file go to the train set'

    def test_load_item(self):
        dataset = HowTo100MData(self.test_dataset_path,
                                self.audio_features_file_name,
                                self.video_features_subdir,
                                self.id_map_file_name,
                                split='train')
        item = dataset[0]
        print({k: item[k].shape for k in item if item[k] is not None})

    # def test_evaluation_has_correct(self):
    #     dataset = HowTo100MData(self.test_dataset_path,
    #                             self.audio_features_file_name,
    #                             self.video_features_subdir,
    #                             self.id_map_file_name,
    #                             split='train')
    #     evaluation_data = dataset.evaluation()
    #     assert 'correct' in evaluation_data

    def setUp(self):
        shutil.copytree(self._dataset_path, self.test_dataset_path)
        # Create a test file
        np.memmap(self.test_dataset_path / self.audio_features_file_name, dtype='float64',
                  mode='w+', shape=(4422, 39))

    def tearDown(self):
        shutil.rmtree(self.test_dataset_path, ignore_errors=False)
