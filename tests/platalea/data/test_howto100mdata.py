import shutil
from pathlib import Path
from unittest import TestCase

import numpy as np

import platalea.dataset


class Howto100mProcessingCase(TestCase):
    _dataset_path = Path('./testdata/howto100msample')  # Should only be accessed by setup.

    test_dataset_path = Path('./testdata/temp_howto100msample')
    id_map_file_name = 'long_id_map.json'
    audio_subdir = 'audio'
    video_subdir = 'video'
    video_features_subdir = 's3d_features'
    audio_features_file_name = 'dummy_mfcc_features.memmap'

    def _get_dataset(self, split):
        dataset = platalea.dataset.HowTo100MData(self.test_dataset_path,
                                                 self.audio_features_file_name,
                                                 self.video_features_subdir,
                                                 self.id_map_file_name,
                                                 split=split)
        return dataset

    def test_get_id_map_number_of_results(self):
        id_map_path = self.test_dataset_path / self.id_map_file_name
        assert len(platalea.dataset._get_id_map(id_map_path, 'test')) == 1000, \
            'By default 1000 ids should go to the test set'

        assert len(platalea.dataset._get_id_map(id_map_path, 'train')) == 3422, \
            'The other 3422 ids in the test file go to the train set'

    def test_len_train(self):
        dataset_train = self._get_dataset('train')
        n_fragments_per_video = 9  # all test videos are 11 seconds long minus 2 seconds for padding
        assert len(dataset_train) == 3422 * n_fragments_per_video, \
            'The other 3422 ids in the test file go to the train set'

    def test_len_test(self):
        dataset_test = self._get_dataset('test')
        n_fragments_per_video = 9  # all test videos are 11 seconds long minus 2 seconds for padding
        assert len(dataset_test) == 1000 * n_fragments_per_video, \
            'By default 1000 ids should go to the test set'

    def test_load_item(self):
        dataset = self._get_dataset('train')
        item = dataset[0]
        print({k: item[k].shape for k in item if item[k] is not None and hasattr(item[k], 'shape')})
        print(item['id'])
        time_steps, feature_size = item['video'].shape
        assert feature_size == 1024
        assert time_steps == dataset.fragment_length

    def test_build_fragment_file_lookup_index(self):
        fragment_length = 3
        video_lengths = np.array([10, 15, 5])
        index = platalea.dataset._fragment_file_lookup_index_from_video_lengths(video_lengths, fragment_length)
        assert np.all(index == np.array([0, 8, 21, 24]))

    def test_get_video_indices_from_fragment_index(self):
        fragment_length = 3
        # index describing a set of 3 files with lengths 10 sec, 15 sec and 5 sec:
        video_lengths = np.array([10, 15, 5])
        fragment_file_lookup_index = platalea.dataset._fragment_file_lookup_index_from_video_lengths(video_lengths,
                                                                                                     fragment_length)

        # first query for the first fragment of file 1:
        fragment_index = 0
        assert (0, 0) == platalea.dataset._get_video_indices_from_fragment_index(fragment_file_lookup_index,
                                                                                 fragment_index, fragment_length)
        # then let's try the last fragment of file 1:
        fragment_index = 7
        assert (0, 7) == platalea.dataset._get_video_indices_from_fragment_index(fragment_file_lookup_index,
                                                                                 fragment_index, fragment_length)
        # first fragment of file 2:
        fragment_index = 8
        assert (1, 0) == platalea.dataset._get_video_indices_from_fragment_index(fragment_file_lookup_index,
                                                                                 fragment_index, fragment_length)
        # last fragment (so, file 3):
        fragment_index = 23
        assert (2, 2) == platalea.dataset._get_video_indices_from_fragment_index(fragment_file_lookup_index,
                                                                                 fragment_index, fragment_length)

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
