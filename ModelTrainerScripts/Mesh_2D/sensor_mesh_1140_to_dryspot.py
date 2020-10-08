from pathlib import Path
import torch
from Pipeline.data_loader_mesh import DataLoaderMesh
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Trainer.ModelTrainer import ModelTrainer
import socket
from Models.erfh5_MeshModel import SensorMeshToDryspotResnet
from Trainer.evaluation import BinaryClassificationEvaluator
import Utils.custom_mlflow
import Resources.training as r
from Utils.mesh_utils import MeshCreator

if __name__ == '__main__':

    debug = False

    if "swt-dgx" in socket.gethostname():
        home_folder = Path("/cfs/home/l/o/lodesluk/code/mesh")
        sensor_verts_path = home_folder / "sensor_verts.dump"
        sample_file = home_folder / "2019-07-24_16-32-40_308_RESULT.erfh5"
        print("Running on DGX")

        filepaths = r.get_data_paths_base_0()
        save_path = r.save_path
        batch_size = 1024
        train_print_frequency = 100
        epochs = 5
        num_workers = 75
        num_validation_samples = 131072
        num_test_samples = 1048576
        data_root = r.data_root
        cache_path = None
        weights_path = home_folder / "checkpoint_sensor2ff.pth"

    elif "pop-os" in socket.gethostname():
        sensor_verts_path = Path("/home/lukas/rtm/sensor_verts.dump")
        sample_file = Path("/home/lukas/rtm/rtm_files/2019-07-24_16-32-40_308_RESULT.erfh5")

        Utils.custom_mlflow.logging = False

        print("Running local mode.")
        base_path = Path("/home/lukas/rtm/")

        filepaths = [base_path / "rtm_files"]
        save_path = Path(base_path / "output")
        batch_size = 64
        train_print_frequency = 50
        epochs = 20
        num_workers = 8
        num_validation_samples = 5000
        num_test_samples = 5000
        data_root = Path(base_path / "rtm_files")
        load_datasets_path = None
        # cache_path = base_path / "cache"
        cache_path = None
        weights_path = Path("/home/lukas/rtm/results/sensor2flow_2020-07-01_16-50-49/checkpoint.pth")
        # weights_path = None
    else:
        print("No valid configuration for this machine. Aborting...")
        exit()

    if "pop-os" in socket.gethostname() and debug:
        print("Debug: True")
        epochs = 1000
        filepaths = [base_path / "debug"]
        batch_size = 4
        num_validation_samples = 4
        num_test_samples = 4
        data_root = Path(base_path / "debug")

    dlm = DataLoaderMesh(sensor_verts_path=sensor_verts_path)
    mc = MeshCreator(batch_size)
    mesh = mc.batched_mesh_torch(batch_size)
    # model = SensorMeshToDryspotModel(mesh, batch_size=batch_size, weights_path=weights_path)
    model = SensorMeshToDryspotResnet(mesh, batch_size=batch_size, weights_path=weights_path)

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
        data_processing_function=dlm.get_sensor_dryspot_mesh,
        data_gather_function=get_filelist_within_folder_blacklisted,
        data_root=data_root,
        loss_criterion=torch.nn.BCELoss(),
        optimizer_function=lambda params: torch.optim.SGD(params, lr=0.001),
        classification_evaluator_function=lambda: BinaryClassificationEvaluator(),
        lr_scheduler_function=None,
        caching_torch=False,
        demo_path=None,
        drop_last_batch=True
    )

    print("Starting training routine.")
    m.start_training()
    print("Training finished. Starting test.")
    m.inference_on_test_set(
        classification_evaluator_function=lambda summary_writer:
        BinaryClassificationEvaluator(save_path / "eval_on_test_set",
                                      skip_images=True,
                                      with_text_overlay=True)
    )
