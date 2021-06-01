import shutil
from pathlib import Path
from unittest import TestCase

from platalea.data.howto100mdata import _get_id_map


class Howto100mProcessingCase(TestCase):
    id_map_path = Path('./testdata/') / 'long_id_map.json'

    def test_get_id_map_number_of_results(self):
        assert len(_get_id_map(self.id_map_path, 'test')) == 1000, \
            'By default 1000 ids should go to the test set'

        assert len(_get_id_map(self.id_map_path, 'train')) == 3422, \
            'The other 3422 ids in the test file go to the train set'
