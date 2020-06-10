from pathlib import Path
import torch
import Resources.training as r
from Models.sensor_to_fiberfraction_model import STFF_v2
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Pipeline.data_loaders_IMG import DataloaderImageSequences
from Trainer.evaluation import SensorToFlowfrontEvaluator
from Utils.training_utils import read_cmd_params
from Trainer.ModelTrainer import ModelTrainer
if __name__ == "__main__":
    args = read_cmd_params()

    batch_size = 128
    dl = DataloaderImageSequences()
    m = ModelTrainer(
        lambda: STFF_v2(),
        r.get_all_data_paths(),
        r.save_path,
        load_datasets_path=r.datasets_dryspots,
        cache_path=r.cache_path,
        batch_size=batch_size,
        epochs=10,
        num_workers=75,
        num_validation_samples=500,
        num_test_samples=500,
        data_processing_function=dl.get_sensor_to_perm_map,
        data_gather_function=get_filelist_within_folder_blacklisted,
        loss_criterion=torch.nn.MSELoss(),
        classification_evaluator_function=lambda summary_writer:
        SensorToFlowfrontEvaluator(summary_writer=summary_writer, skip_images=False, ignore_inp=True, save_path=Path("/cfs/home/s/c/schroeni/Images/SensorFiber/")),
        dummy_epoch=False
    )

    if not args.eval:
        m.start_training()
    else:
        m.inference_on_test_set(
            Path(args.eval),
            Path(args.checkpoint_path),
            lambda summary_writer: SensorToFlowfrontEvaluator(summary_writer=summary_writer,
                save_path=Path(args.eval) / "eval_on_test_set",
                skip_images=False,
		        ignore_inp=True,
            ),
        )
