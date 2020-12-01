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
    num_val = 500
    num_test = 500
    lr = 1e-4
    shrink_factor = 2
    conv_lstm_sizes = [128, 32]
    fc_sizes = [2048, 512, 128]
    run_name = "sf2"
    # train-test-splits reinschauen
    # vllt mal slice_start 0 probieren?

    dl = DataloaderDryspots()
    m = ModelTrainer(
        lambda: SensorToBinaryRunwiseModel(slice_start=1, shrink_factor=shrink_factor, conv_lstm_sizes=conv_lstm_sizes,
                                           fc_sizes=fc_sizes),
        dataset_paths,
        r.save_path,
        dataset_split_path=r.dataset_split,
        cache_path=r.cache_path,
        batch_size=batch_size,
        epochs=15,
        num_workers=num_workers,
        num_validation_samples=num_val,
        num_test_samples=num_test,
        data_processing_function=dl.get_sensor_bool_dryspot_runlevel,
        data_gather_function=get_filelist_within_folder_blacklisted,
        loss_criterion=torch.nn.BCELoss(),
        optimizer_function=lambda params: torch.optim.AdamW(params, lr=lr),
        classification_evaluator_function=lambda: BinaryClassificationEvaluator(skip_images=True),
        # lr_scheduler_function=lambda optim: ExponentialLR(optim, 0.5),
        dummy_epoch=False,
        caching_torch=True,
        run_name=run_name,
        save_in_mlflow_directly=True
    )

    if not args.run_eval:
        m.start_training()
    else:
        m.inference_on_test_set(
            Path(args.eval),
            Path(args.checkpoint_path),
            lambda: BinaryClassificationEvaluator(save_path=Path(args.eval) / "eval_on_test_set", skip_images=False)
        )
