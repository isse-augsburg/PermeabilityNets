import hashlib
import inspect
import json
import logging
import os
import pickle
import re
from pathlib import Path
from time import time

import h5py
import numpy as np
import torch
from scipy import spatial
from torch.utils.data import Sampler

import Resources.training as r


def reshape_to_indeces(_input, ix=((1, 4), (1, 4)), goal=80):
    _input = _input.reshape((-1, 1, 38, 30)).contiguous()
    _input = _input[:, :, ix[0][0]::ix[0][1], ix[1][0]::ix[1][1]].contiguous()
    return _input.reshape(-1, 1, goal).contiguous()


class RandomOverSampler(Sampler):
    """
    Sampler to put more emphasis on the last samples of runs.
    The original size of the dataset is kept. Some samples are dropped.

    Arguments:
        data_source (Dataset): dataset to sample from
        emphasize_after_max_minus (int): index from which starting, counting from the back, the samples are used more
            often
        multiply_by (int): by how much those samples are multiplied in the dataset
    """
    def __init__(self, data_source, emphasize_after_max_minus=80, multiply_by=2):
        self.data_source = data_source
        self.emph = emphasize_after_max_minus
        self.multiply_by = multiply_by

    @property
    def num_samples(self):
        return len(self.data_source[0])

    def double_samples_after_step_x_in_each_run(self):
        logger = logging.getLogger()
        logger.debug("Sampling ...")
        t0 = time()
        indices_lower = []
        indices_higher = []
        for i, aux in enumerate(self.data_source[2]):  # Aux data
            index = aux["ix"]
            _max = aux["max"]
            if index > _max - self.emph:
                indices_higher.append(i)
            else:
                indices_lower.append(i)
        logger.debug(f"Going through data took {t0 - time()}")
        t0 = time()
        count_higher = len(indices_higher)
        # Multiply the number of samples at the back of the runs
        hi = torch.cat([torch.tensor(indices_higher) for x in range(self.multiply_by)])
        hi = hi[torch.randperm(len(hi))]
        logger.debug(f"Random 1 took {t0 - time()}")
        t0 = time()
        # Cast to tensor, randomize
        low = torch.tensor(indices_lower)[torch.randperm(len(indices_lower))]
        logger.debug(f"Random 2 took {t0 - time()}")
        t0 = time()
        low = low[:-count_higher * (self.multiply_by - 1)]
        indizes = torch.cat((hi, low))
        indizes = indizes[torch.randperm(len(indizes))]
        logger.debug(f"Random All took {t0 - time()}")
        return indizes

    def __iter__(self):
        indizes = self.double_samples_after_step_x_in_each_run()
        return iter(indizes)

    def __len__(self):
        return self.num_samples


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def load_mean_std(mean_std_f: Path):
    with open(mean_std_f, "rb") as f:
        mean, std = pickle.load(f)
        mean = np.array(mean)
        std = np.array(std)
    return mean, std


def handle_torch_caching(processing_function, data_source_paths, sampler_func, batch_size, num_val, num_test):
    data_loader_info = processing_function.__self__.__dict__
    data_loader_info["data_processing_function"] = processing_function.__name__
    data_loader_info["data_loader_name"] = processing_function.__self__.__class__.__name__
    data_loader_info["data_loader_source"] = inspect.getsource(processing_function.__self__.__class__)
    data_loader_info["data_source_paths"] = [str(p) for p in data_source_paths]
    data_loader_info["batch_size"] = batch_size
    data_loader_info["num_validation_samples"] = num_val
    data_loader_info["num_test_samples"] = num_test
    if sampler_func is None:
        data_loader_info["sampler"] = ""
    else:
        data_loader_info["sampler"] = sampler_func(None).__dict__
    data_loader_str = str(data_loader_info).encode("utf-8")
    data_loader_hash = hashlib.md5(data_loader_str).hexdigest()
    load_and_save_path = r.datasets_dryspots_torch / data_loader_hash
    # logger = logging.getLogger(__name__)
    # if load_and_save_path.exists():
    #     logger.debug("Existing caches: ")
    #     logger.debug(f"{[x for x in load_and_save_path.iterdir() if x.is_file()]}")
    # else:
    #     logger.debug("No existing caches.")
    load_and_save_path.mkdir(exist_ok=True)

    if (r.datasets_dryspots_torch / "info.json").is_file():
        with open(r.datasets_dryspots_torch / "info.json", "r") as f:
            data = json.load(f)
    else:
        data = {}
    data.update({data_loader_hash: data_loader_info})
    with open(r.datasets_dryspots_torch / "info.json", "w") as f:
        json.dump(data, f, cls=NumpyEncoder)

    return load_and_save_path, data_loader_hash


def extract_sensor_coords(fn: Path, third_dim=False, indices=((0, 1), (0, 1))):
    """
    Extract the sensor coordinates as numpy array from a *d.out file, which exists for every run.
    """
    with fn.open() as f:
        content = f.read()
    sensor_coords = []

    for triple in re.findall(r"-?\d+\.\d+ -?\d+\.\d+ -?\d+\.\d+", content):
        sensor_coords.append([float(e) for e in triple.split(' ')])
    _s_coords = np.array(sensor_coords)
    # Cut off last column (z), since it is filled with 1s anyway
    if not third_dim:
        _s_coords = _s_coords[:, :-1]
    # if indices != ((0, 1), (0, 1)):
    #     _s_coords = _s_coords.reshape(38, 30)
    #     _s_coords = _s_coords[indices[0][0]::indices[0][1], indices[1][0]::indices[1][1]]
    #     _s_coords = _s_coords.flatten()
    return _s_coords


def extract_coords_of_mesh_nodes(fn: Path, normalized=True, third_dim=False):
    """
    Extract the coordinates of the mesh nodes as numpy array from a *RESULT.erfh5 file, which exists for every run.
    """
    with h5py.File(fn, 'r') as f:
        coord_as_np_array = f["post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res"][()]

    if not third_dim:
        # Cut off last column (z), since it is filled with 1s anyway
        coord_as_np_array = coord_as_np_array[:, :-1]

    if normalized:
        coord_as_np_array = normalize_coords(coord_as_np_array)

    return coord_as_np_array


def get_node_propery_at_states_and_indices(f: h5py.File, node_property: str, states: list, indices: list = []):
    if indices == []:
        data = [f["post"]["singlestate"][state]["entityresults"]["NODE"]
                [node_property]["ZONE1_set1"]["erfblock"]["res"][()]
                for state in states]
    else:
        data = [f["post"]["singlestate"][state]["entityresults"]["NODE"]
                [node_property]["ZONE1_set1"]["erfblock"]["res"][()][indices]
                for state in states]
    return data


def preprocess_3dsim_sensors(sensors: list):
    pass


def extract_nearest_mesh_nodes_to_sensors(fn: Path, sensor_indices=((0, 1), (0, 1)), target_size=(38, 30, 2),
                                          third_dim=False, subsampled_nodes=None):
    sensor_coords = extract_sensor_coords(Path(str(fn) + "d.out"))

    if third_dim:
        sensor_coords = sensor_coords[25:]
        sensor_coords = np.append(sensor_coords, [[0, 0]], axis=0)

    sensor_coords = sensor_coords.reshape(target_size)
    sensor_coords = sensor_coords[sensor_indices[0][0]:: sensor_indices[0][1],
                                  sensor_indices[1][0]:: sensor_indices[1][1]]

    sensor_coords = sensor_coords.reshape(-1, 2)

    nodes_coords = extract_coords_of_mesh_nodes(Path(str(fn) + "_RESULT.erfh5"), normalized=False, third_dim=False)
    nodes_coords_3d = extract_coords_of_mesh_nodes(Path(str(fn) + "_RESULT.erfh5"), normalized=False, third_dim=True)

    if subsampled_nodes is not None:
        nodes_coords = nodes_coords[subsampled_nodes]
        nodes_coords_3d = nodes_coords_3d[subsampled_nodes]

    dists_indeces = []
    for sensor in sensor_coords:
        dists_indeces.append(spatial.KDTree(nodes_coords).query(sensor))
        print(f"Sensor at {sensor} -> NN at {nodes_coords_3d[dists_indeces[-1][-1]]}")
    dists, indices = zip(*dists_indeces)
    return np.array(indices)


def scale_coords_lautern(input_coords):
    scaled_coords = (input_coords + 23.25) * 10
    return scaled_coords


def scale_coords_leoben(input_coords):
    scaled_coords = input_coords * 10
    return scaled_coords


def normalize_coords(coords, third_dim=False):
    coords = np.array(coords)
    max_c = np.max(coords[:, 0])
    min_c = np.min(coords[:, 0])
    coords[:, 0] = coords[:, 0] - min_c
    coords[:, 0] = coords[:, 0] / (max_c - min_c)
    max_c = np.max(coords[:, 1])
    min_c = np.min(coords[:, 1])
    coords[:, 1] = coords[:, 1] - min_c
    coords[:, 1] = coords[:, 1] / (max_c - min_c)

    if third_dim:
        coords[:, 2] = coords[:, 2] - min_c
        coords[:, 2] = coords[:, 2] / (max_c - min_c)
    return coords


def change_win_to_unix_path_if_needed(_str):
    if os.name == "unix" and _str.startswith("Y:"):
        _str = _str.replace("\\", "/").replace("Y:", "/cfs/share")
    if os.name == "unix" and _str.startswith("X:"):
        _str = _str.replace("\\", "/").replace("X:", "/cfs/home")
    return _str


def get_folder_of_erfh5(file):
    file_string = str(file)
    file_string = file_string.split(".")[0]
    file_string = file_string.split("_")
    del file_string[-1]
    folder = Path("_".join(file_string))
    return folder


if __name__ == '__main__':
    '''extract_nearest_mesh_nodes_to_sensors(
        Path(r'Y:\\data\\RTM\\Leoben\\sim_output\\2019-07-23_15-38-08_5000p\\0\\2019-07-23_15-38-08_0'))
    '''

    extract_nearest_mesh_nodes_to_sensors(
        Path("/home/lukas/rtm/rtm_files/2019-07-24_16-32-40_308")
    )
