from pathlib import Path
import torch
from Pipeline.data_loader_mesh import DataLoaderMesh
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Trainer.ModelTrainer import ModelTrainer
import socket
from Models.erfh5_DGLMeshModel import SensorMeshToFlowFrontModelDGL
from Trainer.evaluation import FlowFrontMeshEvaluator
import Utils.custom_mlflow
import Resources.training as r
from Utils.mesh_utils import MeshCreator

if __name__ == '__main__':
    sensor_verts_path = Path("/home/lukas/rtm/sensor_verts_3d_v2.dump")
    sample_file = Path("/home/lukas/rtm/rtm_files_3d/2020-08-24_11-20-27_75_RESULT.erfh5")

    debug = False

    if "swt-dgx" in socket.gethostname():
        home_folder = Path("/cfs/home/l/o/lodesluk/code/mesh")
        # TODO EDIT SAMPLE PATHS
        sensor_verts_path = home_folder / "sensor_verts.dump"
        sample_file = home_folder / "2019-07-24_16-32-40_308_RESULT.erfh5"

        base_data_dir = Path("/cfs/home/l/o/lodesluk/Sim_Data/output/with_shapes")
        data_directories = [base_data_dir / "2020-08-24_11-20-27_5000p",
                            base_data_dir / "2020-08-26_22-08-05_5000p"]
        print("Running on DGX")

        filepaths = data_directories
        save_path = r.save_path
        batch_size = 256
        train_print_frequency = 100
        epochs = 5
        num_workers = 75
        num_validation_samples = 2500
        num_test_samples = 2500
        data_root = base_data_dir
        cache_path = None
    elif "pop-os" in socket.gethostname():
        Utils.custom_mlflow.logging = False

        print("Running local mode.")
        base_path = Path("/home/lukas/rtm/")

        filepaths = [base_path / "rtm_files_3d"]
        save_path = Path(base_path / "output")
        batch_size = 12
        train_print_frequency = 100
        epochs = 2
        num_workers = 8
        num_validation_samples = 200
        num_test_samples = 200
        data_root = Path(base_path / "rtm_files_3d")
        load_datasets_path = None
        # cache_path = base_path / "cache"
        cache_path = None
    else:
        print("No valid configuration for this machine. Aborting...")
        exit()

    if "pop-os" in socket.gethostname() and debug:
        filepaths = [base_path / "debug"]
        batch_size = 4
        num_validation_samples = 4
        num_test_samples = 4
        data_root = Path(base_path / "debug")

    dlm = DataLoaderMesh(sensor_verts_path=sensor_verts_path)
    mc = MeshCreator(sample_file)
    mesh = mc.batched_mesh_dgl(batch_size)
    model = SensorMeshToFlowFrontModelDGL(mesh, batch_size=batch_size)

    m = ModelTrainer(
        lambda: model,
        data_source_paths=filepaths,
        save_path=save_path,
        cache_path=cache_path,
        batch_size=batch_size,
        train_print_frequency=train_print_frequency,
        epochs=epochs,
        dummy_epoch=True,
        num_workers=num_workers,
        num_validation_samples=num_validation_samples,
        num_test_samples=num_test_samples,
        data_processing_function=dlm.get_sensor_flowfront_mesh,
        data_gather_function=get_filelist_within_folder_blacklisted,
        data_root=data_root,
        loss_criterion=torch.nn.MSELoss(),
        optimizer_function=lambda params: torch.optim.AdamW(params, lr=1e-4),
        classification_evaluator_function=lambda:
        FlowFrontMeshEvaluator(sample_file=sample_file,
                               save_path=save_path / "FF_Images/FF_4290_deeper2"),
        lr_scheduler_function=None,
        caching_torch=False,
        demo_path=None,
        drop_last_batch=True
    )

    m.start_training()
    print("Training finished. Starting evaluation on test set.")
    m.inference_on_test_set(classification_evaluator_function=lambda summary_writer:
                            FlowFrontMeshEvaluator(sample_file=sample_file,
                                                   save_path=save_path / "FF_Images/FF_4290_eval_deeper2"))
