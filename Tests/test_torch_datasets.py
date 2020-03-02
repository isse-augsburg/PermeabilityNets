import logging
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

import Pipeline.data_gather as dg
import Resources.testing as test_resources
import Resources.training as tr_resources
from Models.erfh5_fullyConnected import S20DryspotModelFCWide
from Pipeline.data_loader_flowfront_sensor import DataloaderFlowfrontSensor
from Trainer.ModelTrainer import ModelTrainer
from Trainer.evaluation import BinaryClassificationEvaluator
from Utils.data_utils import change_win_to_unix_path_if_needed


class TestSaveDatasetsTorch(unittest.TestCase):
    def setUp(self) -> None:
        self.torch_dataset_resources = test_resources.torch_datasets
        self.torch_all_datasets = self.torch_dataset_resources / "all"
        self.reference_datasets_torch = tr_resources.datasets_dryspots_torch
        # Hash of the dataloader and datalprocessing_function of these tests
        self.reference_hash = "71cf3ef38ca8a742922d7cf6bb8ac2a0"
        self.load_and_save_path = self.reference_datasets_torch / self.reference_hash

    def create_trainer_and_start(self, out_path, epochs=1):
        dlds = DataloaderFlowfrontSensor(sensor_indizes=((1, 8), (1, 8)))
        m = ModelTrainer(lambda: S20DryspotModelFCWide(),
                         data_source_paths=tr_resources.get_data_paths_debug(),
                         save_path=out_path,
                         load_datasets_path=self.torch_dataset_resources / "reference_datasets",
                         cache_path=None,
                         num_validation_samples=8,
                         num_test_samples=8,
                         num_workers=0,
                         epochs=epochs,
                         data_processing_function=dlds.get_flowfront_sensor_bool_dryspot,
                         data_gather_function=dg.get_filelist_within_folder_blacklisted,
                         loss_criterion=torch.nn.BCELoss(),
                         optimizer_function=lambda params: torch.optim.AdamW(params, lr=1e-4),
                         classification_evaluator_function=lambda summary_writer:
                         BinaryClassificationEvaluator(summary_writer=summary_writer),
                         )
        m.start_training()
        return m

    def compare_old_new_dataset(self, old_data: list, new_data: list):
        for i in range(len(old_data)):
            odata, olabel, osourcepath = old_data[i]
            ndata, nlabel, nsourcepath = new_data[i]
            self.assertTrue(torch.equal(odata, ndata))
            self.assertTrue(torch.equal(olabel, nlabel))
            osourcepath = change_win_to_unix_path_if_needed(osourcepath["sourcefile"])
            self.assertEqual(osourcepath, nsourcepath["sourcefile"])

    def copy_list_of_files_to_load_and_save_path(self, files: list):
        self.load_and_save_path.mkdir(exist_ok=True)
        for file in files:
            shutil.copy(file, self.load_and_save_path)

    def test_save_train_val_test_sets(self):
        with tempfile.TemporaryDirectory(prefix="TorchDataSetsSaving") as tempdir:
            out_path = Path(tempdir)
            m = self.create_trainer_and_start(out_path, epochs=2)

            self.assertTrue((self.load_and_save_path / "train_set_torch.p").is_file())
            self.assertTrue((self.load_and_save_path / "val_set_torch.p").is_file())
            m.inference_on_test_set()
            self.assertTrue((self.load_and_save_path / "test_set_torch.p").is_file())
            logging.shutdown()

    def test_load_all_data_sets(self):
        self.copy_list_of_files_to_load_and_save_path([self.torch_all_datasets / "train_set_torch.p",
                                                       self.torch_all_datasets / "val_set_torch.p",
                                                       self.torch_all_datasets / "test_set_torch.p"])
        with tempfile.TemporaryDirectory(prefix="TorchDataSetsLoading") as tempdir:
            m = self.create_trainer_and_start(Path(tempdir))
            self.assertTrue(m.data_generator.loaded_train_set)
            self.assertTrue(m.data_generator.loaded_val_set)
            self.assertTrue(m.data_generator.loaded_train_set)
            logging.shutdown()

    def test_load_train_set_only(self):
        """ Check if the splits are produced the same way if only the training set is loaded and if the the
            loaded_* flags are set correctly.
        """
        self.copy_list_of_files_to_load_and_save_path([self.torch_all_datasets / "train_set_torch.p"])
        with tempfile.TemporaryDirectory(prefix="TorchDataSetsLoading") as tempdir:
            m = self.create_trainer_and_start(Path(tempdir))
            self.assertTrue(m.data_generator.loaded_train_set)
            self.assertFalse(m.data_generator.loaded_val_set)
            self.assertFalse(m.data_generator.loaded_test_set)
            with open(self.torch_dataset_resources / "all/val_set_torch.p", "rb") as f:
                saved_val_set = torch.load(f)
            with open(self.torch_dataset_resources / "all/test_set_torch.p", "rb") as f:
                saved_test_set = torch.load(f)
            self.compare_old_new_dataset(saved_val_set, m.data_generator.get_validation_samples())
            self.compare_old_new_dataset(saved_test_set, m.data_generator.get_test_samples())
            logging.shutdown()

    def test_load_train_and_test_set_only(self):
        self.copy_list_of_files_to_load_and_save_path([self.torch_all_datasets / "train_set_torch.p",
                                                       self.torch_all_datasets / "test_set_torch.p"])
        with tempfile.TemporaryDirectory(prefix="TorchDataSetsLoading") as tempdir:
            m = self.create_trainer_and_start(Path(tempdir))
            self.assertTrue(m.data_generator.loaded_train_set)
            self.assertFalse(m.data_generator.loaded_val_set)
            self.assertTrue(m.data_generator.loaded_test_set)
            with open(self.torch_dataset_resources / "all/test_set_torch.p", "rb") as f:
                saved_test_set = torch.load(f)
            self.compare_old_new_dataset(saved_test_set, m.data_generator.get_test_samples())
            logging.shutdown()

    def test_load_train_and_val_set_only(self):
        self.copy_list_of_files_to_load_and_save_path([self.torch_all_datasets / "train_set_torch.p",
                                                       self.torch_all_datasets / "val_set_torch.p"])
        with tempfile.TemporaryDirectory(prefix="TorchDataSetsLoading") as tempdir:
            m = self.create_trainer_and_start(Path(tempdir))
            self.assertTrue(m.data_generator.loaded_train_set)
            self.assertTrue(m.data_generator.loaded_val_set)
            self.assertFalse(m.data_generator.loaded_test_set)
            with open(self.torch_dataset_resources / "all/val_set_torch.p", "rb") as f:
                saved_val_set = torch.load(f)
            self.compare_old_new_dataset(saved_val_set, m.data_generator.get_validation_samples())
            logging.shutdown()

    def test_load_val_and_test_set_only(self):
        """
        Load the val and test set only and check if the flags are set correctly, and the training set
        is loaded correctly
        """
        self.copy_list_of_files_to_load_and_save_path([self.torch_all_datasets / "val_set_torch.p",
                                                       self.torch_all_datasets / "test_set_torch.p"])
        with tempfile.TemporaryDirectory(prefix="TorchDataSetsLoading") as tempdir:
            m = self.create_trainer_and_start(Path(tempdir))
            self.assertFalse(m.data_generator.loaded_train_set)
            self.assertTrue(m.data_generator.loaded_val_set)
            self.assertTrue(m.data_generator.loaded_test_set)
            with open(self.torch_dataset_resources / "all/train_set_torch.p", "rb") as f:
                saved_train_set = torch.load(f)
            old_paths = set([change_win_to_unix_path_if_needed(x[2]["sourcefile"]) for x in saved_train_set])
            new_paths = set([x[2]["sourcefile"][0] for x in m.data_generator])
            self.assertEqual(old_paths, new_paths)
            logging.shutdown()

    def tearDown(self) -> None:
        logging.shutdown()
        if self.load_and_save_path.exists():
            shutil.rmtree(self.load_and_save_path)