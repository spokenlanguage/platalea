import filecmp
import os
import shutil
from pathlib import Path
from unittest import TestCase

from platalea.utils.preprocessing import preprocess_howto100m


class Howto100mProcessingCase(TestCase):
    _dataset_path = Path('./testdata/howto100msample')  # Should only be accessed by setup.
    test_dataset_path = Path('./testdata/temp_howto100msample')
    audio_subdir = 'audio'
    video_subdir = 'video'
    video_features_subdir = 's3d_features'

    def do_preprocess_call(self):
        preprocess_howto100m(self.test_dataset_path,
                             self.audio_subdir,
                             self.video_subdir,
                             self.video_features_subdir)

    def test_it_runs(self):
        self.do_preprocess_call()

    def test_no_files_changed(self):
        self.do_preprocess_call()
        assert_has_at_least_same_unchanged_content(self._dataset_path, self.test_dataset_path)

    def test_audio_dir_created(self):
        self.do_preprocess_call()

        audio_dir = self.test_dataset_path / self.audio_subdir
        assert audio_dir.exists()

    def test_audio_files_created(self):
        self.do_preprocess_call()

        audio_dir = self.test_dataset_path / self.audio_subdir
        video_dir = self.test_dataset_path / self.video_subdir
        assert len(os.listdir(audio_dir)) == len(os.listdir(video_dir))

    def test_audio_features_created(self):
        self.do_preprocess_call()
        features_file_path = self.test_dataset_path / 'mfcc_features.memmap'
        assert features_file_path.exists()

    def test_id_map_created(self):
        self.do_preprocess_call()
        id_map_file_path = self.test_dataset_path / 'id_map.json'
        assert id_map_file_path.exists()

    def setUp(self):
        shutil.copytree(self._dataset_path, self.test_dataset_path)

    def tearDown(self):
        shutil.rmtree(self.test_dataset_path, ignore_errors=False)


def assert_has_at_least_same_unchanged_content(source_dir, target_dir):
    """Returns True if all files in the source dir are present and unchanged in the target dir.

    Returns false otherwise.
    NB, files in the target dir that are not in the source dir, are ignored.
    """
    for entry in os.listdir(source_dir):
        current_source_path = source_dir / entry
        current_target_path = target_dir / entry
        if current_source_path.is_dir():
            assert_has_at_least_same_unchanged_content(current_source_path, current_target_path)
        else:
            if not filecmp.cmp(current_source_path, current_target_path):
                raise AssertionError('Change found in {}'.format(current_source_path))
