from pathlib import Path
import torch
from torch.optim.lr_scheduler import ExponentialLR

import Resources.training as r
from Models.transfer_learning_models import ModelWrapper
from Pipeline.data_gather import get_filelist_within_folder_blacklisted, get_filelist_within_folder
from Pipeline.data_loader_dryspot import DataloaderDryspots
from Trainer.ModelTrainer import ModelTrainer
from Trainer.evaluation import BinaryClassificationEvaluator
from Utils.training_utils import read_cmd_params
from torchvision import models 
import socket

if __name__ == "__main__":

    args = read_cmd_params()

    print(">>> Model: Inception v3")

    model = ModelWrapper(models.resnext50_32x4d(pretrained=True))
    #model = ModelWrapper(models.inception_v3(pretrained=True, aux_logits=False))

    if "swt-dgx" in socket.gethostname():
        print("On DGX. - Using Inception")
        filepaths = r.get_data_paths_base_0()
        save_path = r.save_path
        batch_size = 512
        train_print_frequency = 100
        epochs = 50
        num_workers = 75
        num_validation_samples = 512
        num_test_samples = 512
        data_gather_function = get_filelist_within_folder_blacklisted
        data_root = r.data_root
    else: 
        print("Running local mode.")
        filepaths = [Path("H:/RTM Files/LocalDebug/")]
        save_path = Path("H:/RTM Files/output")
        batch_size = 4
        train_print_frequency = 5
        epochs = 5
        num_workers = 8
        num_validation_samples = 2
        num_test_samples = 2
        data_gather_function = get_filelist_within_folder
        data_root = Path("H:/RTM Files/LocalDebug/")

    dlds = DataloaderDryspots()
    m = ModelTrainer(
        lambda: model,
        data_source_paths=filepaths,
        save_path=save_path,
        load_datasets_path=None,
        cache_path=None,
        batch_size=batch_size,
        train_print_frequency=train_print_frequency,
        epochs=epochs,
        dummy_epoch=True,
        num_workers=num_workers,
        num_validation_samples=num_validation_samples,
        num_test_samples=num_test_samples,
        data_processing_function=dlds.get_sensor_bool_dryspot_299x299,
        data_gather_function=get_filelist_within_folder_blacklisted,
        data_root=data_root,
        loss_criterion=torch.nn.BCELoss(),
        optimizer_function=lambda params: torch.optim.AdamW(params, lr=1e-4),
        classification_evaluator_function=lambda summary_writer:
        BinaryClassificationEvaluator(summary_writer=summary_writer),
        lr_scheduler_function=lambda optim: ExponentialLR(optim, 0.5),
        caching_torch=False,
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
