from pathlib import Path
import torch
from Pipeline.data_loader_mesh import DataLoaderMesh
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Trainer.ModelTrainer import ModelTrainer
import socket
from Models.erfh5_MeshModel import SensorMeshToFlowFrontModel
from Trainer.evaluation import MeshEvaluator
import Utils.custom_mlflow
from Utils.mesh_utils import MeshCreator

if __name__ == '__main__':
    # sensor_verts_path = Path("/home/lukas/rtm/sensor_verts.dump")
    sensor_verts_path = None
    sample_file = Path("/home/lukas/rtm/rtm_files/2019-07-24_16-32-40_308_RESULT.erfh5")

    Utils.custom_mlflow.logging = False
    debug = True

    if "swt-dgx" in socket.gethostname():
        pass
    elif "pop-os" in socket.gethostname():
        print("Running local mode.")
        base_path = Path("/home/lukas/rtm/")

        filepaths = [base_path / "rtm_files"]
        save_path = Path(base_path / "output")
        batch_size = 96
        train_print_frequency = 100
        epochs = 5
        num_workers = 8
        num_validation_samples = 5000
        num_test_samples = 5000
        data_root = Path(base_path / "rtm_files")
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
    mc = MeshCreator(batch_size)
    mesh = mc.batched_mesh_torch(batch_size)
    model = SensorMeshToFlowFrontModel(mesh, batch_size=batch_size)

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
        loss_criterion=torch.nn.BCELoss(),
        optimizer_function=lambda params: torch.optim.AdamW(params, lr=1e-5),
        classification_evaluator_function=lambda: MeshEvaluator(),
        lr_scheduler_function=None,
        caching_torch=False,
        demo_path=None,
        drop_last_batch=True
    )

    m.start_training()
