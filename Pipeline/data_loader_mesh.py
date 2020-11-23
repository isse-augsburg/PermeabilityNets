import h5py
import numpy as np
# import open3d
import logging
from pathlib import Path
from Utils.data_utils import normalize_coords, extract_nearest_mesh_nodes_to_sensors, \
    get_folder_of_erfh5
import pickle
import os


class DataLoaderMesh:
    """Data Loader for training directly on meshes. The functions extract features for each node.

    Args:
        divide_by_100k (Bool): Should the data processing functions divide the sensor values by 100 000?
        sensor_verts_path (Path-like): Path to a pickle dump that contains the indices of all vertices in the mesh that
                                       are nearest neighbors to a sensor
        ignore_useless_states (Bool): Should useless states for dryspot detection be ignored?
        sensor_indices: Used to increase distance between sensors. E.g. ((1, 4), (1, 4)) means every 4th sensor is used.
        third_dim (Bool): Should internal functions like extract_nearest_mesh_nodes_to_sensors use the 3rd dim of
                          coordinates?
        intermediate_target_size (triple): Intermediate shape of sensor grid used for calculating the final sensors.
    """
    def __init__(self, divide_by_100k=False,
                 sensor_verts_path=None,
                 ignore_useless_states=False,
                 sensor_indices=((0, 1), (0, 1)),
                 third_dim=True,
                 intermediate_target_size=(66, 65, 2)
                 ):

        self.divide_by_100k = divide_by_100k
        self.sensor_verts = None
        self.ignore_useless_states = ignore_useless_states
        self.sensor_indices = sensor_indices
        self.third_dim = third_dim
        self.intermediate_target_size = intermediate_target_size

        self.sensor_verts_path = sensor_verts_path

        self.subsampled_nodes = None

        if (self.sensor_verts_path is not None) and (os.path.isfile(self.sensor_verts_path)):
            logger = logging.getLogger()
            logger.info('Loading sensor vertices from pickle file.')
            self.sensor_verts = pickle.load(open(sensor_verts_path, 'rb'))
            if self.third_dim:
                self.sensor_verts = self.sensor_verts[:-1]
            print(f'Loaded {len(self.sensor_verts)} sensor vertices.')

    def get_sensor_flowfront_mesh(self, filename):
        """Returns samples of shape (num_vertices, 1)  with following values:
           if nearest neighbor of sensor: sensorvalue else: 0
           The label is of shape (num_vertices, 1) containing the filling factor on each node.
        """
        f = h5py.File(filename, 'r')
        folder = get_folder_of_erfh5(filename)
        instances = []

        if self.sensor_verts is None:
            print("Calculating sensor vertices from scratch.")
            self.sensor_verts = extract_nearest_mesh_nodes_to_sensors(folder, sensor_indices=self.sensor_indices,
                                                                      target_size=self.intermediate_target_size,
                                                                      third_dim=self.third_dim,
                                                                      subsampled_nodes=self.subsampled_nodes)
            if self.sensor_verts_path is not None:
                pickle.dump(self.sensor_verts, open(self.sensor_verts_path, 'wb'))
                print(f"Saved sensor vertices in {self.sensor_verts_path}.")

            if self.third_dim:
                self.sensor_verts = self.sensor_verts[:-1]
            print(f"Calculated {len(self.sensor_verts)} sensor vertices.")

        try:
            verts = f["post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/"
                      "erfblock/res"][()]

            if self.subsampled_nodes is not None:
                verts = verts[self.subsampled_nodes]

            verts = normalize_coords(verts, third_dim=True)

            states = f["post"]["singlestate"]
            all_inputs = []
            all_labels = []

            # Get all pressure and filling factor states
            for s in states:
                input_features = np.zeros((verts.shape[0]))
                pressure = f['post']['singlestate'][s]['entityresults']['NODE'][
                    'PRESSURE']['ZONE1_set1']['erfblock']['res'][()]

                flowfront = f['post']['singlestate'][s]['entityresults']['NODE'][
                    'FILLING_FACTOR']['ZONE1_set1']['erfblock']['res'][()]

                flowfront = np.squeeze(np.ceil(flowfront))

                if self.subsampled_nodes is not None:
                    pressure = pressure[self.subsampled_nodes]
                    flowfront = flowfront[self.subsampled_nodes]

                input_features[self.sensor_verts] = np.squeeze(pressure[self.sensor_verts])
                if self.divide_by_100k:
                    # input_features = input_features / 100000
                    input_features = input_features * 10
                all_inputs.append(input_features)
                all_labels.append(flowfront)

            for i, x in enumerate(all_inputs):
                instances.append((x, all_labels[i]))

            f.close()
            return instances

        except KeyError:
            logger = logging.getLogger()
            logger.warning(f'KeyError: {filename}')
            f.close()
            return None

    def get_sensor_dryspot_mesh(self, filename):
        """Returns samples of shape (num_vertices, 1)  with following values:
           if nearest neighbor of sensor: sensorvalue else: 0
           The label is either 0 or 1, whether the sample contains a dryspot or not.
        """

        f = h5py.File(filename, 'r')
        meta_file = h5py.File(str(filename).replace("RESULT.erfh5", "meta_data.hdf5"), 'r')
        folder = get_folder_of_erfh5(filename)
        instances = []
        all_labels = []

        if self.sensor_verts is None:
            print("calculating sensor vertices from scratch.")
            self.sensor_verts = extract_nearest_mesh_nodes_to_sensors(folder)
            print("Calculated sensor vertices.")

        try:
            verts = f["post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/"
                      "erfblock/res"][()]
            verts = normalize_coords(verts, third_dim=True)

            array_of_states = meta_file["dryspot_states/singlestates"][()]
            if self.ignore_useless_states:
                useless_states = meta_file["useless_states/singlestates"][()]
            set_of_states = set(array_of_states.flatten())

            states = f["post"]["singlestate"]
            all_inputs = []

            # Get all pressure states and labels
            for s in states:
                if self.ignore_useless_states and len(useless_states) > 0 and s == f'state{useless_states[0]:012d}':
                    break

                input_features = np.zeros((verts.shape[0]))
                pressure = f['post']['singlestate'][s]['entityresults']['NODE'][
                    'PRESSURE']['ZONE1_set1']['erfblock']['res'][()]

                input_features[self.sensor_verts] = np.squeeze(pressure[self.sensor_verts])
                if self.divide_by_100k:
                    input_features = input_features / 100000
                all_inputs.append(input_features)

                label = 0
                state_num = int(str(s).replace("state", "0"))
                if state_num in set_of_states:
                    label = 1

                all_labels.append(label)

            for i, x in enumerate(all_inputs):
                instances.append((x, all_labels[i]))

            f.close()
            return instances

        except KeyError:
            logger = logging.getLogger()
            logger.warning(f'KeyError: {filename}')
            f.close()
            return None


if __name__ == '__main__':
    sensor_verts_path = Path("/home/lukas/rtm/sensor_verts.dump")
    dl = DataLoaderMesh()
    # file = Path("/home/lukas/rtm/rtm_files/2019-07-24_16-32-40_308_RESULT.erfh5")
    file = Path("/home/lukas/rtm/rtm_files_3d/2020-08-24_11-20-27_111_RESULT.erfh5")
    # instances = dl.get_sensor_flowfront_mesh(file)
    # instances = dl.get_sensor_dryspot_mesh(file)
