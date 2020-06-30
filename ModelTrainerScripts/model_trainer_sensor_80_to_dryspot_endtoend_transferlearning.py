from pathlib import Path
import torch
from torch.optim.lr_scheduler import ExponentialLR

import Resources.training as r
from Models.erfh5_ConvModel import S80Deconv2ToDrySpotTransferLearning
from Pipeline.data_gather import get_filelist_within_folder_blacklisted, \
    get_filelist_within_folder
from Pipeline.data_loader_dryspot import DataloaderDryspots
from Trainer.ModelTrainer import ModelTrainer
from Trainer.evaluation import BinaryClassificationEvaluator
from Utils.training_utils import read_cmd_params
from Pipeline.TorchDataGeneratorUtils.looping_strategies \
    import NoOpLoopingStrategy
import socket

if __name__ == "__main__":

    args = read_cmd_params()

    eval_on_test = False

    if "swt-dgx" in socket.gethostname():
        print("On DGX. - 80 Sensors. Resnext as backbone.")
        filepaths = r.get_data_paths_base_0()
        save_path = r.save_path
        batch_size = 1024
        train_print_frequency = 100
        epochs = 5
        num_workers = 75
        num_validation_samples = 131072
        num_test_samples = 1048576
        # num_test_samples = 524288
        data_gather_function = get_filelist_within_folder_blacklisted
        data_root = r.data_root
        cache_path = r.cache_path

        weights = Path("/cfs/home/s/t/stiebesi/data/RTM/"
                       "Results/IJCAI_PRICAI_20_FlowFrontNet/"
                       "S80_to_DS_deconv_conv/"
                       "2020-02-28_12-07-38_deconv_second_try/checkpoint.pth")
        pretrained = "deconv_weights"
        freeze_nlayers = 9

    else:
        print("Running local mode.")

        basepath = Path("/home/lukas/rtm/rtm_files")
        filepaths = [basepath]
        save_path = Path("/home/lukas/rtm/output/")
        batch_size = 2
        train_print_frequency = 100
        epochs = 5
        num_workers = 8
        num_validation_samples = 8000
        num_test_samples = 8000
        data_gather_function = get_filelist_within_folder
        data_root = basepath
        load_datasets_path = None
        cache_path = None

        weights = None
        pretrained = False
        freeze_nlayers = 0

    model = S80Deconv2ToDrySpotTransferLearning(pretrained=pretrained,
                                                checkpoint_path=weights,
                                                freeze_nlayers=freeze_nlayers
                                                )

    def init_trainer():

        dlds = DataloaderDryspots(sensor_indizes=((1, 4), (1, 4)))
        m = ModelTrainer(
            lambda: model,
            data_source_paths=filepaths,
            save_path=save_path,
            cache_path=cache_path,
            batch_size=batch_size,
            train_print_frequency=train_print_frequency,
            looping_strategy=NoOpLoopingStrategy(),
            epochs=epochs,
            dummy_epoch=False,
            num_workers=num_workers,
            num_validation_samples=num_validation_samples,
            num_test_samples=num_test_samples,
            data_processing_function=dlds.get_sensor_bool_dryspot,
            data_gather_function=get_filelist_within_folder_blacklisted,
            data_root=data_root,
            loss_criterion=torch.nn.BCELoss(),
            optimizer_function=lambda params: torch.optim.AdamW(
                params, lr=1e-5),
            classification_evaluator_function=lambda summary_writer:
            BinaryClassificationEvaluator(summary_writer=summary_writer),
            lr_scheduler_function=lambda optim: ExponentialLR(optim, 0.5),
            caching_torch=False,
            demo_path=None,
            hold_in_ram=False,
        )

        return m

    print("Starting training.")
    m = init_trainer()
    m.start_training()
    print("Training finished. Starting evaluation")
    m.inference_on_test_set(
        classification_evaluator_function=lambda summary_writer:
        BinaryClassificationEvaluator(save_path / "eval_on_test_set",
                                      skip_images=True,
                                      with_text_overlay=True)
    )
