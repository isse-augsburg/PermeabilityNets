from pathlib import Path

import torch

import Resources.training as r
from Models.erfh5_fullyConnected import S80DryspotModelFCWide
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Pipeline.data_loader_dryspot import DataloaderDryspots
from Trainer.ModelTrainer import ModelTrainer
from Trainer.evaluation import BinaryClassificationEvaluator
from Utils.training_utils import read_cmd_params

if __name__ == "__main__":
    """
    This is the starting point for training the feed foward network with 80 sensor data to binary classification.
    """
    args = read_cmd_params()

    dlds = DataloaderDryspots(sensor_indizes=((1, 4), (1, 4)))
    m = ModelTrainer(
        lambda: S80DryspotModelFCWide(demo_mode=True if args.demo is not None else False),
        data_source_paths=r.get_data_paths_base_0(),
        save_path=r.save_path if args.demo is None else Path(args.demo),
        dataset_split_path=r.dataset_split,
        cache_path=r.cache_path,
        batch_size=32768,
        train_print_frequency=100,
        epochs=100,
        num_workers=75,
        num_validation_samples=131072,
        num_test_samples=1048576,
        data_processing_function=dlds.get_sensor_bool_dryspot,
        data_gather_function=get_filelist_within_folder_blacklisted,
        loss_criterion=torch.nn.BCELoss(),
        optimizer_function=lambda params: torch.optim.AdamW(params, lr=1e-4),
        classification_evaluator_function=lambda: BinaryClassificationEvaluator(),
        demo_path=args.demo
    )

    if not args.run_eval:
        m.start_training()
    else:
        m.inference_on_test_set(
            output_path=Path(args.eval),
            checkpoint_path=Path(args.checkpoint_path),
            classification_evaluator_function=lambda: BinaryClassificationEvaluator(
                Path(args.eval) / "eval_on_test_set",
                skip_images=True,
                with_text_overlay=True)
        )
