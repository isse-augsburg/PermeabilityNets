from pathlib import Path

import torch
from torch.optim.lr_scheduler import ExponentialLR

import Resources.training as r
from Models.erfh5_ConvModel import SensorDeconvToDryspotEfficient2
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Pipeline.data_loader_dryspot import DataloaderDryspots
from Trainer.ModelTrainer import ModelTrainer
from Trainer.evaluation import BinaryClassificationEvaluator
from Utils.training_utils import read_cmd_params

if __name__ == "__main__":
    """
    Producing data only. Using to sequential sampler to unshuffle the data + 1 "thread" only to make sure there is no shuffling between threads.
    """
    dlds = DataloaderDryspots()
    m = ModelTrainer(
        lambda: SensorDeconvToDryspotEfficient2(),
        data_source_paths=r.get_data_paths_base_0(),
        save_path=r.save_path,
        load_datasets_path=r.datasets_dryspots,
        cache_path=r.cache_path,
        batch_size=2048,
        num_workers=1,
        num_validation_samples=131072,
        num_test_samples=1048576,
        data_processing_function=dlds.get_sensor_bool_dryspot,
        data_gather_function=get_filelist_within_folder_blacklisted,
        classification_evaluator_function=lambda summary_writer:
        BinaryClassificationEvaluator(summary_writer=summary_writer),
        produce_torch_datasets_only=True,
        sampler=lambda data_source: torch.utils.data.SequentialSampler(data_source=data_source)
    )

    m.start_training()
