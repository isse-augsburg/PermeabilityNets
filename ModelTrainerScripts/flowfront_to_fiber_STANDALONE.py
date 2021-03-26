from Models.flowfrontPermBaseline import FF2Perm_Baseline, FF2Perm_3DConv
from Models.flowfront2PermTransformer import OptimusPrime, OptimusPrime2
from pathlib import Path
import socket
import torch
import numpy as np
import Resources.training as r
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
    save_path = Path(r"C:\Users\schroeni\Documents\Projekte\COSIMO\DATA\OUT")
    demo_path = Path(r"C:\Users\schroeni\Documents\Projekte\COSIMO\DATA\FlowfrontToPermeability")

    dl = DataloaderImageSequences()
    m = ModelTrainer(
        lambda: OptimusPrime2(batch_size),
        dataset_paths,
        save_path,
        cache_path=r.cache_path,
        batch_size=batch_size,
        epochs=150,
        num_workers=num_workers,
        num_validation_samples=num_val,
        num_test_samples=num_test,
        data_processing_function=dl.get_flowfront_to_perm_map,
        data_gather_function=get_filelist_within_folder_blacklisted,
        loss_criterion=torch.nn.MSELoss(),
        data_root=data_root,
        demo_path=demo_path,
        classification_evaluator_function=lambda: SensorToFlowfrontEvaluator(skip_images=False, ignore_inp=False,
                                                                             sensors_shape=(143, 111),
                                                                             save_path=save_path),
        dummy_epoch=False
    )

    if not args.eval:
        m.start_training()
    else:
        m.inference_on_test_set(
            Path(args.eval),
            Path(args.checkpoint_path),
            lambda summary_writer: SensorToFlowfrontEvaluator(save_path=Path(args.eval) / "eval_on_test_set",
                                                              skip_images=False,
                                                              ignore_inp=True,
                                                              sensors_shape=(143, 111)),
        )
