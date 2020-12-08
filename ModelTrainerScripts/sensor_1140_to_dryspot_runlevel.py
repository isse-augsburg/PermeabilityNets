from pathlib import Path

import torch

import Resources.training as r
from Models.sensor_to_binary_run_model import SensorToBinaryRunwiseModel
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Pipeline.data_loader_dryspot import DataloaderDryspots
from Utils.dicts.sensor_dicts import sensor_indices
from Trainer.ModelTrainer import ModelTrainer
from Trainer.evaluation import BinaryClassificationEvaluator
from Utils.training_utils import read_cmd_params

if __name__ == "__main__":
    args = read_cmd_params()

    batch_size = 128
    data_root = Path('/cfs/share/data/RTM/Leoben/sim_output_every_step')  # r.data_root
    dataset_paths = r.get_regular_sampled_data_paths()  # r.get_all_data_paths()
    num_workers = 75
    use_cache = False
    num_val = 50
    num_test = 50
    num_epochs = 15
    lr = 1e-4
    num_sensors = 1140  # {1140, 285, 80, 20}
    conv_lstm_sizes = [128, 32]
    fc_sizes = [2048, 512, 128]
    create_data_plots = True
    run_name = "new data (split path None)"

    dl = DataloaderDryspots(sensor_indizes=sensor_indices[str(num_sensors)], aux_info=True, image_size=(143, 111))
    m = ModelTrainer(
        lambda: SensorToBinaryRunwiseModel(num_sensors, conv_lstm_sizes, fc_sizes),
        dataset_paths,
        r.save_path,
        dataset_split_path=None,  # r.dataset_split,
        data_root=data_root,
        cache_path=r.cache_path,
        batch_size=batch_size,
        epochs=num_epochs,
        num_workers=num_workers,
        num_validation_samples=num_val,
        num_test_samples=num_test,
        data_processing_function=dl.get_sensor_bool_dryspot_runlevel,
        data_gather_function=get_filelist_within_folder_blacklisted,
        loss_criterion=torch.nn.BCELoss(),
        optimizer_function=lambda params: torch.optim.AdamW(params, lr=lr),
        classification_evaluator_function=lambda: BinaryClassificationEvaluator(skip_images=not create_data_plots,
                                                                                max_epochs=num_epochs, data_loader=dl),
        # lr_scheduler_function=lambda optim: ExponentialLR(optim, 0.5),
        dummy_epoch=False,
        caching_torch=use_cache,
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
