from pathlib import Path
import torch
import Resources.training as r
from Models.sensor_to_fiberfraction_model import STFF
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Pipeline.data_loaders_IMG import DataloaderImageSequences
from Trainer.evaluation import SensorToFlowfrontEvaluator
from Utils.training_utils import read_cmd_params
from Trainer.ModelTrainer import ModelTrainer
if __name__ == "__main__":
    args = read_cmd_params()

    batch_size = 8
    dl = DataloaderImageSequences()
    m = ModelTrainer(
        lambda: STFF(),
        r.get_data_paths(),
        r.save_path,
        load_datasets_path=r.datasets_dryspots,
        cache_path=r.cache_path,
        batch_size=batch_size,
        epochs=5,
        num_workers=8,
        num_validation_samples=10,
        num_test_samples=10,
        data_processing_function=dl.get_sensor_to_perm_map,
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
