from Models.flowfrontPermBaseline import FF2Perm_Baseline, FF2Perm_3DConv
from Models.flowfront2PermTransformer import OptimusPrime, OptimusPrime2, OptimusPrime_c2D
from pathlib import Path
import socket
import torch
import numpy as np
from Models.sensor_to_fiberfraction_model import AttentionFFTFF, FFTFF, ThreeDAttentionFFTFF
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Pipeline.data_loaders_IMG import DataloaderImageSequences
from Trainer.evaluation import SensorToFlowfrontEvaluator
from Utils.training_utils import read_cmd_params_attention, read_cmd_params
from Trainer.ModelTrainer import ModelTrainer
from Utils import custom_mlflow
if __name__ == "__main__":
    args = read_cmd_params()
    custom_mlflow.logging = False
    batch_size = 8
    dataset_paths = []
    num_workers = 20
    num_val = 2
    num_test = 1
    data_root = Path("")

    #### CONFIGURATION ####

    # Folder where output will be saved
    save_path = Path(r"C:\Users\schroeni\Documents\Projekte\COSIMO\DATA\OUT")
    # Folder containing the three sets .pt
    data_folder = Path(r"C:\Users\schroeni\Documents\Projekte\COSIMO\DATA\FlowfrontToPermeability")
    # Folder containing the model checkpoints
    model_chkpts = Path(r"C:\Users\schroeni\Documents\Projekte\COSIMO\DATA\Checkpoints")
    # choose between Conv2D, Conv3D, Transformer, or ConvLSTM. It will run ConvLSTM when something wrong is specified
    mode = 'Transformer'
    # set Eval to False if you want to run the training
    eval = True

    dl = DataloaderImageSequences()
    m = ModelTrainer(
        lambda: OptimusPrime_c2D(batch_size) if mode == "Transformer" else (FFTFF() if mode == "ConvLSTM" else (FF2Perm_Baseline() if mode == "Conv2D" else (FF2Perm_3DConv() if mode == "Conv3D" else FFTFF()))),
        dataset_paths,
        save_path,
        cache_path=None,
        batch_size=batch_size,
        epochs=150,
        num_workers=num_workers,
        num_validation_samples=num_val,
        num_test_samples=num_test,
        data_processing_function=dl.get_flowfront_to_perm_map,
        data_gather_function=get_filelist_within_folder_blacklisted,
        loss_criterion=torch.nn.MSELoss(),
        data_root=data_root,
        demo_path=data_folder,
        classification_evaluator_function=lambda: SensorToFlowfrontEvaluator(skip_images=False, ignore_inp=False,
                                                                             sensors_shape=(143, 111),
                                                                             save_path=save_path),
        dummy_epoch=False
    )

    if not eval:
        m.start_training()
    else:
        m.inference_on_test_set(
            Path(save_path) / Path(mode),
            Path(model_chkpts) / ("checkpoint" + mode + ".pth"),
            lambda: SensorToFlowfrontEvaluator(save_path=Path(save_path) / Path(mode) / "eval_on_test_set",
                                                              skip_images=False,
                                                              ignore_inp=True,
                                                              sensors_shape=(143, 111)),
        )
