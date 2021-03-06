from pathlib import Path

import torch

import Resources.training as r
from Models.erfh5_DeconvModel import DeconvModelEfficient
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Pipeline.data_loaders_IMG import DataloaderImages
from Trainer.ModelTrainer import ModelTrainer
from Trainer.evaluation import SensorToFlowfrontEvaluator
from Utils.training_utils import read_cmd_params

if __name__ == "__main__":
    """
    This is the starting point for training the Deconv/Conv Part of the FlowFrontNet 
    with 1140 sensor data to Flowfront images.
    """
    args = read_cmd_params()

    dl = DataloaderImages(image_size=(149, 117))
    m = ModelTrainer(
        lambda: DeconvModelEfficient(),
        data_source_paths=r.get_data_paths_base_0(),
        save_path=r.save_path if args.demo is None else Path(args.demo),
        dataset_split_path=r.dataset_split,
        cache_path=r.cache_path,
        batch_size=2048,
        train_print_frequency=100,
        epochs=1000,
        num_workers=75,
        num_validation_samples=131072,
        num_test_samples=1048576,
        data_processing_function=dl.get_sensordata_and_flowfront,
        data_gather_function=get_filelist_within_folder_blacklisted,
        loss_criterion=torch.nn.MSELoss(),
        optimizer_function=lambda params: torch.optim.AdamW(params, lr=1e-4),
        classification_evaluator_function=lambda: SensorToFlowfrontEvaluator(),
        demo_path=args.demo,
        # run_eval_step_before_training=True
    )

    if not args.run_eval:
        m.start_training()
    else:
        m.inference_on_test_set(
            Path(args.eval),
            Path(args.checkpoint_path),
            lambda: SensorToFlowfrontEvaluator(Path(args.eval) / "eval_on_test_set", skip_images=False)
        )
