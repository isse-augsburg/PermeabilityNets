from pathlib import Path

import torch

import Resources.training as r
from Models.flow_front_to_fiber_fraction_model import FlowfrontToFiberfractionModel
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Pipeline.data_loaders_IMG import DataloaderImageSequences
from Trainer.ModelTrainer import ModelTrainer
from Trainer.evaluation import SensorToFlowfrontEvaluator
from Utils.training_utils import read_cmd_params

if __name__ == "__main__":
    args = read_cmd_params()

    batch_size = 8
    dl = DataloaderImageSequences(image_size=(149, 117), wanted_frames=10)

    m = ModelTrainer(
        lambda: FlowfrontToFiberfractionModel(),
        r.get_data_paths(),
        r.save_path,
        dataset_split_path=r.dataset_split,
        cache_path=r.cache_path,
        batch_size=batch_size,
        epochs=5,
        num_workers=8,
        num_validation_samples=10,
        num_test_samples=10,
        data_processing_function=dl.get_images_of_flow_front_and_permeability_map,
        data_gather_function=get_filelist_within_folder_blacklisted,
        loss_criterion=torch.nn.MSELoss(),
        classification_evaluator_function=lambda summary_writer:
        SensorToFlowfrontEvaluator(summary_writer=summary_writer),
        dummy_epoch=False
    )

    if not args.eval:
        m.start_training()
    else:
        m.inference_on_test_set(
            Path(args.eval_path),
            Path(args.checkpoint_path),
            SensorToFlowfrontEvaluator(
                Path(args.eval_path) / "eval_on_test_set",
                skip_images=False,
            ),
        )
