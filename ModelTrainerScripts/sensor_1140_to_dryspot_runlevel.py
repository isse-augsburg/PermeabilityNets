from pathlib import Path

import torch

import Resources.training as r
from Models.sensor_to_binary_run_model import SensorToBinaryRunwiseModel
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Pipeline.data_loader_dryspot import DataloaderDryspots
from Trainer.ModelTrainer import ModelTrainer
from Trainer.evaluation import BinaryClassificationEvaluator
from Utils.training_utils import read_cmd_params

if __name__ == "__main__":
    args = read_cmd_params()

    batch_size = 128
    dataset_paths = r.get_all_data_paths()
    num_workers = 75
    num_val =  1000
    num_test = 1000

    dl = DataloaderDryspots()
    m = ModelTrainer(
        lambda: SensorToBinaryRunwiseModel(slice_start=1, shrink_factor=8),
        dataset_paths,
        r.save_path,
        dataset_split_path=r.dataset_split,
        cache_path=r.cache_path,
        batch_size=batch_size,
        epochs=10,
        num_workers=num_workers,
        num_validation_samples=num_val,
        num_test_samples=num_test,
        data_processing_function=dl.get_sensor_bool_dryspot_runlevel,
        data_gather_function=get_filelist_within_folder_blacklisted,
        loss_criterion=torch.nn.BCELoss(),
        classification_evaluator_function=lambda summary_writer:
        BinaryClassificationEvaluator(summary_writer=summary_writer, skip_images=True,
                                      save_path=Path("/cfs/home/s/e/sertolbe/sensor-to-dryspot-runlevel/")),
        dummy_epoch=False
    )

    if not args.eval:
        m.start_training()
    else:
        m.inference_on_test_set(
            Path(args.eval),
            Path(args.checkpoint_path),
            lambda summary_writer: BinaryClassificationEvaluator(summary_writer=summary_writer,
                                                                 save_path=Path(args.eval) / "eval_on_test_set",
                                                                 skip_images=False,
                                                                 ),
        )
