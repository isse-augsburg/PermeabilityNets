import h5py
import numpy as np
import logging
from pathlib import Path
from pytorch3d.structures import Meshes
import torch
from Utils.data_utils import normalize_coords, extract_nearest_mesh_nodes_to_sensors, \
    get_folder_of_erfh5
import pickle


class DataLoaderMesh:
    def __init__(self, divide_by_100k=True,
                 sensor_verts_path=None):

        #TODO use it
        self.divide_by_100k = divide_by_100k
        self.sensor_verts = None

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
            print("calculating sensor vertices from scratch.")
            self.sensor_verts = extract_nearest_mesh_nodes_to_sensors(folder)
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
        except:
            logger = logging.getLogger()
            logger.warning(f'Exception: {filename}')
            f.close()
            return None

    def get_batched_mesh(self, batchsize, filename):
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
            faces = faces.repeat(batchsize, 1, 1)
            verts = torch.unsqueeze(torch.tensor(verts), dim=0)
            verts = verts.repeat(batchsize, 1, 1)
            mesh = Meshes(verts=verts, faces=faces)

            f.close()
            return mesh

        except KeyError:
            logger = logging.getLogger()
            logger.warning(f'Error: Calculation of mesh failed.')
            f.close()
            return None


if __name__ == '__main__':
    dl = DataLoaderMesh(sensor_verts_path=Path("/home/lukas/rtm/sensor_verts.dump"))
    file = Path("/home/lukas/rtm/rtm_files/2019-07-24_16-32-40_308_RESULT.erfh5")
    # mesh = dl.get_batched_mesh(4, file)
    instances = dl.get_sensor_flowfront_mesh(file)
    pass
