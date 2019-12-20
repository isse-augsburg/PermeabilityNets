import logging
import re
import shutil
import unittest

import Tests.resources_for_testing as resources
from model_trainer_dryspot import DrySpotTrainer


class TestTrainingDryspotFF(unittest.TestCase):
    def setUp(self):
        self.training_save_path = resources.test_training_out_dir
        self.training_data_paths = [resources.test_training_src_dir / 'dry_spot_from_ff']
        self.expected_num_epochs_during_training = 1
        self.dt = DrySpotTrainer(
            data_source_paths=self.training_data_paths,
            batch_size=10,
            eval_freq=1,
            save_datasets_path=self.training_save_path,
            epochs=self.expected_num_epochs_during_training,
            num_validation_samples=5,
            num_test_samples=5,
        )

    def test_training(self):
        self.dt.run_training()
        dirs = [e for e in self.training_save_path.iterdir() if e.is_dir()]
        with open(dirs[0] / "output.log") as f:
            content = f.read()
            epochs = re.findall("Mean Loss on Eval", content)
            self.assertTrue(len(epochs) > 0)

    def tearDown(self) -> None:
        logging.shutdown()
        r = logging.getLogger("")
        [r.removeHandler(x) for x in r.handlers]
        shutil.rmtree(self.training_save_path)


if __name__ == "__main__":
    unittest.main()