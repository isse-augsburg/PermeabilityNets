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
from Pipeline.TorchDataGeneratorUtils.looping_strategies import NoOpLoopingStrategy
from torchvision import models 
import socket

if __name__ == "__main__":

    args = read_cmd_params()

    print(">>> Model: Resnext")

    model = ModelWrapper(models.resnext50_32x4d(pretrained=True))
    #model = ModelWrapper(models.inception_v3(pretrained=True, aux_logits=False))

    if "swt-dgx" in socket.gethostname():
        print("On DGX. - Using ResNeXt. More frames. No caching in ram. BS 1024")
        filepaths = r.get_data_paths_base_0()
        save_path = r.save_path
        batch_size = 1024
        train_print_frequency = 100
        epochs = 2
        num_workers = 75
        num_validation_samples = 131072
        num_test_samples = 1048576
        data_gather_function = get_filelist_within_folder_blacklisted
        data_root = r.data_root
        load_datasets_path=r.datasets_dryspots
        cache_path = r.cache_path
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
        load_datasets_path=None
        cache_path = None

    dlds = DataloaderDryspots()
    m = ModelTrainer(
        lambda: model,
        data_source_paths=filepaths,
        save_path=save_path,
        load_datasets_path=load_datasets_path,
        cache_path=cache_path,
        batch_size=batch_size,
        train_print_frequency=train_print_frequency,
        looping_strategy=NoOpLoopingStrategy(),
        epochs=epochs,
        dummy_epoch=True,
        num_workers=num_workers,
        num_validation_samples=num_validation_samples,
        num_test_samples=num_test_samples,
        data_processing_function=dlds.get_sensor_bool_dryspot_resized_matrix,
        data_gather_function=get_filelist_within_folder_blacklisted,
        data_root=data_root,
        loss_criterion=torch.nn.BCELoss(),
        optimizer_function=lambda params: torch.optim.AdamW(params, lr=1e-4),
        classification_evaluator_function=lambda summary_writer:
        BinaryClassificationEvaluator(summary_writer=summary_writer),
        lr_scheduler_function=lambda optim: ExponentialLR(optim, 0.5),
        caching_torch=False,
        demo_path=None,
        hold_in_ram=False,
    )

    print("Starting training.")
    m.start_training()
    print("Training done. Starting evaluation on test set.")

    m.inference_on_test_set(
        classification_evaluator_function=lambda summary_writer:
        BinaryClassificationEvaluator(save_path / "eval_on_test_set",
                                        skip_images=True,
                                        with_text_overlay=True)
    )
