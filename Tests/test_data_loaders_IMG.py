import logging
import shutil
import unittest

import Tests.resources_for_testing as Resources
from Pipeline.data_loaders_IMG import \
    get_images_of_flow_front_and_permeability_map
from Pipeline.resampling import get_fixed_number_of_indices


class TestDataLoaderIMG(unittest.TestCase):
    def setUp(self):
        self.img_cache_dirname = Resources.data_loader_img_file

    # @unittest.skip("Currently not working")
    def test_get_fixed_number_of_elements_and_their_indices_from_various_sized_list(
            self):
        for i in [2, 10, 20, 33, 100]:
            for j in [2, 5, 8, 10, 20]:
                self.create_list_and_test(i, j)

    def create_list_and_test(self, list_length, n_elements):
        logger = logging.getLogger(__name__)
        logger.info(list_length, n_elements)
        if n_elements > list_length:
            return
        x = get_fixed_number_of_indices(
            list_length, n_elements)

        self.assertEqual(len(x), n_elements)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
