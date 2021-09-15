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

    def _get_dataset(self, split):
        dataset = HowTo100MData(self.test_dataset_path,
                                self.audio_features_file_name,
                                self.video_features_subdir,
                                self.id_map_file_name,
                                split=split)
        return dataset

    def test_get_id_map_number_of_results(self):
        id_map_path = self.test_dataset_path / self.id_map_file_name
        assert len(_get_id_map(id_map_path, 'test')) == 1000, \
            'By default 1000 ids should go to the test set'

        assert len(_get_id_map(id_map_path, 'train')) == 3422, \
            'The other 3422 ids in the test file go to the train set'

    def test_load_item(self):
        dataset = self._get_dataset('train')
        item = dataset[0]
        print({k: item[k].shape for k in item if item[k] is not None and hasattr(item[k], 'shape')})
        print(item['id'])

    def test_evaluation_has_video(self):
        dataset = self._get_dataset('train')
        evaluation_data = dataset.evaluation()
        assert 'video' in evaluation_data

    def test_evaluation_has_audio(self):
        dataset = self._get_dataset('train')
        evaluation_data = dataset.evaluation()
        assert 'audio' in evaluation_data

    def test_evaluation_has_same_number_of_audio_video(self):
        dataset = self._get_dataset('test')
        evaluation_data = dataset.evaluation()
        assert len(evaluation_data['audio']) == len(evaluation_data['video'])

    def setUp(self):
        self._remove_temp_test_data_from_earlier_runs()

        shutil.copytree(self._dataset_path, self.test_dataset_path)
        # Create a test file
        np.memmap(self.test_dataset_path / self.audio_features_file_name, dtype='float64',
                  mode='w+', shape=(4422, 39))

    def _remove_temp_test_data_from_earlier_runs(self):
        """ Use of memmap makes cleanup during teardown unreliable.
            This function is just for convenience in case last teardown failed.
        """
        if self.test_dataset_path.exists():
            shutil.rmtree(self.test_dataset_path, ignore_errors=False)

    def tearDown(self):
        shutil.rmtree(self.test_dataset_path, ignore_errors=False)
