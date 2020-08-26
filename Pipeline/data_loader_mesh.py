import h5py
import numpy as np
import logging
from pathlib import Path
from pytorch3d.structures import Meshes
import torch
from Utils.data_utils import normalize_coords, extract_nearest_mesh_nodes_to_sensors, \
    get_folder_of_erfh5
import pickle
import dgl


class DataLoaderMesh:
    def __init__(self, divide_by_100k=False,
                 sensor_verts_path=None,
                 ignore_useless_states=False,
                 sensor_indices=((0, 1), (0, 1))
                 ):

        self.divide_by_100k = divide_by_100k
        self.sensor_verts = None
        self.ignore_useless_states = ignore_useless_states
        self.sensor_indices = sensor_indices

        if sensor_verts_path is not None:
            logger = logging.getLogger()
            logger.info('Loading sensor vertices from pickle file.')
            self.sensor_verts = pickle.load(open(sensor_verts_path, 'rb'))
            logger.info('Loaded sensor vertices.')

    def get_sensor_flowfront_mesh(self, filename):
        f = h5py.File(filename, 'r')
        folder = get_folder_of_erfh5(filename)
        instances = []

        if self.sensor_verts is None:
            print("Calculating sensor vertices from scratch.")
            self.sensor_verts = extract_nearest_mesh_nodes_to_sensors(folder, sensor_indices=self.sensor_indices)
            print("Calculated sensor vertices.")

        try:
            verts = f["post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/"
                      "erfblock/res"][()]
            verts = normalize_coords(verts)

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
            verts = normalize_coords(verts)

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

    def __get_mesh_components(self, filename):
        f = h5py.File(filename, 'r')

        try:
            verts = f["post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/"
                      "erfblock/res"][()]
            verts = normalize_coords(verts)
            # Get internal indices of nodes
            hashes = f["post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/"
                       "erfblock/entid"][()]
            hashes = {h: i for i, h in enumerate(hashes)}

            # Calculate faces based on internal indices
            faces = f["post/constant/connectivities/SHELL/erfblock/ic"][()]
            faces = faces[:, :-1]
            faces = np.vectorize(hashes.__getitem__)(faces)

            faces = torch.unsqueeze(torch.Tensor(faces), dim=0)
            verts = torch.unsqueeze(torch.tensor(verts), dim=0)

            f.close()
            return verts, faces

        except KeyError:
            logger = logging.getLogger()
            logger.warning(f'KeyError: Calculation of mesh failed.')
            f.close()
            raise Exception('Calculation of mesh failed because of a KeyError')
        except IndexError:
            logger = logging.getLogger()
            logger.warning(f'KeyError: Calculation of mesh failed.')
            f.close()
            raise Exception('Calculation of mesh failed because of a IndexError')

    def get_batched_mesh_torch(self, batchsize, filename):

        verts, faces = self.__get_mesh_components(filename)

        faces = faces.repeat(batchsize, 1, 1)
        verts = verts.repeat(batchsize, 1, 1)
        mesh = Meshes(verts=verts, faces=faces)

        return mesh

    def get_batched_mesh_dgl(self, batchsize, filename):

        verts, faces = self.__get_mesh_components(filename)

        # Using PyTorch3d Meshes because of its transformations, e.g. edges_packed()
        mesh = Meshes(verts=verts, faces=faces)
        edges = mesh.edges_packed().numpy()
        src, dst = np.split(edges, 2, axis=1)
        u = np.squeeze(np.concatenate([src, dst]))
        v = np.squeeze(np.concatenate([dst, src]))
        dgl_mesh = dgl.DGLGraph((u, v))
        batch = [dgl_mesh for i in range(batchsize)]
        batched_mesh = dgl.batch(batch)
        return batched_mesh




if __name__ == '__main__':
    sensor_verts_path = Path("/home/lukas/rtm/sensor_verts.dump")
    dl = DataLoaderMesh()
    file = Path("/home/lukas/rtm/rtm_files/2019-07-24_16-32-40_308_RESULT.erfh5")
    # mesh = dl.get_batched_mesh(4, file)
    instances = dl.get_sensor_flowfront_mesh(file)
    # instances = dl.get_sensor_dryspot_mesh(file)
    pass
