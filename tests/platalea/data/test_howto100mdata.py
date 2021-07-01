from pathlib import Path
from unittest import TestCase

from platalea.data.howto100mdata import _get_id_map, HowTo100MData


class Howto100mProcessingCase(TestCase):
    id_map_path = Path('./testdata/') / 'long_id_map.json'

    def test_get_id_map_number_of_results(self):
        assert len(_get_id_map(self.id_map_path, 'test')) == 1000, \
            'By default 1000 ids should go to the test set'

        assert len(_get_id_map(self.id_map_path, 'train')) == 3422, \
            'The other 3422 ids in the test file go to the train set'

    def test_load_item(self):
        dataset_path = Path('./testdata/howto100msample')
        id_map_path = r'C:\Users\ChristiaanMeijer\Documents\spoken-language\platalea\testdata\long_id_map.json'
        audio_features_path = r'C:\Users\ChristiaanMeijer\Documents\spoken-language\platalea\testdata\howto100msample\mfcc_features.memmap'
        video_features_subdir = 's3d_features'
        dataset = HowTo100MData(dataset_path, audio_features_path, video_features_subdir,  id_map_path, split='train')
        item = dataset[0]
        print({k: item[k].shape for k in item if item[k] is not None})
