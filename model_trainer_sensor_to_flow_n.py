from pathlib import Path
import torch
import Resources.training as r
from Models.erfh5_DeconvModel import DeconvModelEfficient
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Pipeline.data_loaders_IMG import DataloaderImages
from Trainer.evaluation import SensorToFlowfrontEvaluator
from Utils.training_utils import read_cmd_params
from Trainer.GenericTrainer import ModelTrainer

if __name__ == "__main__":
    args = read_cmd_params()

    batch_size = 4
    dl = DataloaderImages((149, 117))

    m = ModelTrainer(
        lambda: DeconvModelEfficient(),
        r.get_data_paths(),
        r.save_path,
        load_datasets_path=r.datasets_dryspots,
        cache_path=r.cache_path,
        batch_size=batch_size,
        train_print_frequency=10,
        epochs=5,
        num_workers=8,
        num_validation_samples=10,
        num_test_samples=10,
        data_processing_function=dl.get_sensordata_and_flowfront,
        data_gather_function=get_filelist_within_folder_blacklisted,
        loss_criterion=torch.nn.MSELoss(),
        learning_rate=0.0001,
        classification_evaluator=SensorToFlowfrontEvaluator(),
    )

    if not args.eval:
        m.start_training()
    else:
        m.inference_on_test_set(
            Path(args.eval_path),
            Path(args.checkpoint_path),
            SensorToFlowfrontEvaluator(
                Path(args.eval_path) / "eval_on_test_set", skip_images=False
            ),
        )
