from pathlib import Path

import torch
from torch.optim.lr_scheduler import ExponentialLR

import Resources.training as r
from Models.erfh5_ConvModel import S80Deconv2ToDrySpotEff
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Pipeline.data_loader_dryspot import DataloaderDryspots
from Trainer.ModelTrainer import ModelTrainer
from Trainer.evaluation import BinaryClassificationEvaluator
from Utils.training_utils import read_cmd_params

if __name__ == "__main__":
    """
    This is the second stage for training the FlowFrontNet: the DrySpotNet 
    with 80 sensor data to Flowfront images.
    Please add the path to the pretrained weights as parameter 
    checkpoint_path to the SensorDeconvToDryspotEfficient2 Model or use the command line option --checkpoint_path.
    """
    args = read_cmd_params()
    if args.demo is not None:
        checkpoint_path = args.checkpoint_path
    else:
        checkpoint_path = "Use your own path to checkpoint."

    dl = DataloaderDryspots(sensor_indizes=((1, 4), (1, 4)))

    m = ModelTrainer(
        lambda: S80Deconv2ToDrySpotEff(demo_mode=True if args.demo is not None else False,
                                       pretrained="deconv_weights",
                                       checkpoint_path=Path(checkpoint_path),
                                       freeze_nlayers=9),
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
        data_processing_function=dl.get_sensor_bool_dryspot,
        data_gather_function=get_filelist_within_folder_blacklisted,
        loss_criterion=torch.nn.MSELoss(),
        optimizer_function=lambda params: torch.optim.AdamW(params, lr=0.0001),
        classification_evaluator_function=lambda: BinaryClassificationEvaluator(),
        lr_scheduler_function=lambda optim: ExponentialLR(optim, 0.5),
        demo_path=args.demo
    )

    if not args.run_eval:
        m.start_training()
    else:
        m.inference_on_test_set(
            Path(args.eval),
            Path(args.checkpoint_path),
            lambda: BinaryClassificationEvaluator(Path(args.eval) / "eval_on_test_set"),
        )
