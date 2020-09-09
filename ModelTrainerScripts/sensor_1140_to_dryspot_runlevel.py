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
    dataset_paths = r.get_data_paths()
    load_from_cache = True
    num_workers = 75
    num_val = 500
    num_test = 500
    lr = 1e-4
    run_name = ""

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
        optimizer_function=lambda params: torch.optim.AdamW(params, lr=lr),
        classification_evaluator_function=lambda summary_writer:
        BinaryClassificationEvaluator(summary_writer=summary_writer, skip_images=True),
        dummy_epoch=False,
        caching_torch=load_from_cache,
        run_name=run_name,
        save_in_mlflow_directly=False
    )

    if not args.run_eval:
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
