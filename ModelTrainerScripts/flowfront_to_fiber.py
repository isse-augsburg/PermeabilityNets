from Models.flowfrontPermBaseline import FF2Perm_Baseline, FF2Perm_3DConv
from Models.flowfront2PermTransformer import OptimusPrime, OptimusPrime2, OptimusPrime_c2D
from pathlib import Path
import socket
import torch
import numpy as np
import Resources.training as r
from Models.sensor_to_fiberfraction_model import AttentionFFTFF, FFTFF, ThreeDAttentionFFTFF
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Pipeline.data_loaders_IMG import DataloaderImageSequences
from Trainer.evaluation import SensorToFlowfrontEvaluator
from Utils.training_utils import read_cmd_params_attention
from Trainer.ModelTrainer import ModelTrainer
if __name__ == "__main__":
    args = read_cmd_params_attention()
    

    if "swt-dgx" in socket.gethostname():
        batch_size = 32
        dataset_paths = r.get_regular_sampled_data_paths()
        num_workers = 35
        num_val = 100
        num_test = 400
        data_root = r.data_root_every_step
        img_save_path = Path("/cfs/home/s/c/schroeni/Images/FlowFrontToFiber/TransPretrained")
        chpkt = r"/cfs/share/cache/output_schroeni/2021-02-17_14-54-44/checkpoint.pth"
    else:
        batch_size = 8
        dataset_paths = r.get_data_paths_debug()
        num_workers = 20
        num_val = 2
        num_test = 1
        data_root = r.data_root
        img_save_path = Path(r"C:\Users\schroeni\CACHE\Saved_Imgs\FFtoPerm")
        chpkt = r"X:\cache\output_schroeni\2021-02-17_14-54-44\checkpoint.pth"

    dl = DataloaderImageSequences()
    m = ModelTrainer(
        lambda: OptimusPrime_c2D(batch_size),
        dataset_paths,
        r.save_path,
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
        classification_evaluator_function=lambda: SensorToFlowfrontEvaluator(skip_images=False, ignore_inp=False,
                                                                             sensors_shape=(143, 111),
                                                                             save_path=img_save_path),
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
