from Models.flowfront2PermTransformer import OptimusPrime
from pathlib import Path
import socket
import torch
import numpy as np
import Resources.training as r
# from Models.sensor_to_fiberfraction_model import AttentionFFTFF, FFTFF, ThreeDAttentionFFTFF
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Pipeline.data_loaders_IMG import DataloaderImageSequences
from Trainer.evaluation import SensorToFlowfrontEvaluator
from Utils.training_utils import read_cmd_params_attention
from Trainer.ModelTrainer import ModelTrainer
if __name__ == "__main__":
    args = read_cmd_params_attention()
    torch.manual_seed(0)
    np.random.seed(0)

    if "swt-dgx" in socket.gethostname():
        batch_size = 32
        dataset_paths = r.get_regular_sampled_data_paths()
        num_workers = 35
        num_val = 100
        num_test = 400
        data_root = r.data_root_every_step
    else:
        batch_size = 2
        dataset_paths = r.get_data_paths_debug()
        num_workers = 6
        num_val = 2
        num_test = 1
        data_root = r.data_root

    dl = DataloaderImageSequences()
    m = ModelTrainer(
        lambda: OptimusPrime(batch_size),
        dataset_paths,
        r.save_path,
        cache_path=r.cache_path,
        batch_size=batch_size,
        epochs=50,
        num_workers=num_workers,
        num_validation_samples=num_val,
        num_test_samples=num_test,
        data_processing_function=dl.get_flowfront_to_perm_map,
        data_gather_function=get_filelist_within_folder_blacklisted,
        loss_criterion=torch.nn.MSELoss(),
        data_root=data_root,
        classification_evaluator_function=lambda: SensorToFlowfrontEvaluator(skip_images=False, ignore_inp=False,
                                                                             sensors_shape=(143, 111),
                                                                             save_path=Path(
                                                                                 "/cfs/home/s/c/schroeni/Images"
                                                                                 "/FlowFrontToFiber/")),
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
