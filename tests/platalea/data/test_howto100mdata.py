import shutil

from pathlib import Path
from time import sleep
from unittest import TestCase

from platalea.data.howto100mdata import _get_id_map, HowTo100MData
import numpy as np

class Howto100mProcessingCase(TestCase):
    _dataset_path = Path('./testdata/howto100msample')  # Should only be accessed by setup.


    test_dataset_path = Path('./testdata/temp_howto100msample')
    id_map_file_name = 'long_id_map.json'
    audio_subdir = 'audio'
    video_subdir = 'video'
    video_features_subdir = 's3d_features'
    audio_features_path = 'mfcc_features.memmap'
    dataset = None


    def test_get_id_map_number_of_results(self):
        id_map_path = self.test_dataset_path / self.id_map_file_name
        assert len(_get_id_map(id_map_path, 'test')) == 1000, \
            'By default 1000 ids should go to the test set'

        assert len(_get_id_map(id_map_path, 'train')) == 3422, \
            'The other 3422 ids in the test file go to the train set'

    def test_load_item(self):
        self.dataset = HowTo100MData(self.test_dataset_path,
                                self.audio_features_path,
                                self.video_features_subdir,
                                self.id_map_file_name,
                                split='train')
        # item = self.dataset[0]
        # print({k: item[k].shape for k in item if item[k] is not None})
        # now copy the memmap to a bigger one and use that one.

    def setUp(self):
        shutil.copytree(self._dataset_path, self.test_dataset_path)

    def tearDown(self):
        if self.dataset:
            self.dataset.audio.flush()
            self.dataset = None  # Remove references to unlink the memmap
        shutil.rmtree(self.test_dataset_path, ignore_errors=False)