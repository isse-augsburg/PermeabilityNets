from pathlib import Path
import torch
from torch.optim.lr_scheduler import ExponentialLR

import Resources.training as r
from Models.erfh5_ConvModel import SensorDeconvToDryspotEfficient2
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Pipeline.data_loader_dryspot import DataloaderDryspots
from Trainer.ModelTrainer import ModelTrainer
from Trainer.evaluation import BinaryClassificationEvaluator
from Utils.training_utils import read_cmd_params
from torchvision import models 

if __name__ == "__main__":

    args = read_cmd_params()


    print(">>> Model: Resnet18")
    print("Caching deactivated")
    model = models.resnet18(pretrained=True)
    # model = models.resnext50_32x4d(pretrained=True)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 1)

    dlds = DataloaderDryspots()
    m = ModelTrainer(
        lambda: model,
        data_source_paths=r.get_data_paths_base_0(),
        save_path=r.save_path,
        load_datasets_path=None,
        cache_path=None,
        batch_size=2048,
        train_print_frequency=100,
        epochs=1000,
        num_workers=75,
        num_validation_samples=131072,
        num_test_samples=1048576,
        data_processing_function=dlds.get_sensor_bool_dryspot_299x299,
        data_gather_function=get_filelist_within_folder_blacklisted,
        loss_criterion=torch.nn.BCELoss(),
        optimizer_function=lambda params: torch.optim.AdamW(params, lr=1e-4),
        classification_evaluator_function=lambda summary_writer:
        BinaryClassificationEvaluator(summary_writer=summary_writer),
        lr_scheduler_function=lambda optim: ExponentialLR(optim, 0.5),
        demo_path=None
    )

    if not args.run_eval:
        m.start_training()
    else:
        m.inference_on_test_set(
            output_path=Path(args.eval),
            checkpoint_path=Path(args.checkpoint_path),
            classification_evaluator_function=lambda summary_writer:
            BinaryClassificationEvaluator(Path(args.eval) / "eval_on_test_set",
                                          skip_images=True,
                                          with_text_overlay=True)
        )
