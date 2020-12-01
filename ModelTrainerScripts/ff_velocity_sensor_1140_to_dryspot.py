from pathlib import Path

import torch

import Resources.training as r
from Models.erfh5_fullyConnected import S1140DryspotModelFCWide
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Pipeline.data_loader_flowfront_sensor import DataloaderFlowfrontSensor
from Trainer.ModelTrainer import ModelTrainer
from Trainer.evaluation import BinaryClassificationEvaluator
from Utils.training_utils import read_cmd_params

if __name__ == "__main__":
    args = read_cmd_params()

    dlds = DataloaderFlowfrontSensor(sensor_indizes=((0, 1), (0, 1)),
                                     frame_count=1,
                                     use_binary_sensor_only=True)
    m = ModelTrainer(lambda: S1140DryspotModelFCWide(),
                     data_source_paths=r.get_data_paths_base_0(),
                     save_path=r.save_path,
                     dataset_split_path=r.dataset_split,
                     cache_path=r.cache_path,
                     batch_size=2048,
                     train_print_frequency=100,
                     epochs=1000,
                     num_workers=75,
                     num_validation_samples=131072,
                     num_test_samples=1048576,
                     data_processing_function=dlds.get_flowfront_sensor_bool_dryspot,
                     data_gather_function=get_filelist_within_folder_blacklisted,
                     loss_criterion=torch.nn.BCELoss(),
                     optimizer_function=lambda params: torch.optim.AdamW(params, lr=1e-4),
                     classification_evaluator_function=lambda: BinaryClassificationEvaluator(),
                     dont_care_num_samples=True
                     # lr_scheduler_function=lambda optim: ExponentialLR(optim, 0.1),
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
