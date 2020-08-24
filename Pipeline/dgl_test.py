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


def get_batched_mesh_dgl(batchsize, filename):
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
        # faces = faces.repeat(batchsize, 1, 1)
        verts = torch.unsqueeze(torch.tensor(verts), dim=0)
        # verts = verts.repeat(batchsize, 1, 1)

        mesh = Meshes(verts=verts, faces=faces)
        edges = mesh.edges_packed().numpy()
        src, dst = np.split(edges, 2, axis=1)
        u = np.squeeze(np.concatenate([src, dst]))
        v = np.squeeze(np.concatenate([dst, src]))
        dgl_mesh = dgl.DGLGraph((u, v))
        f.close()
        return mesh

    except KeyError:
        logger = logging.getLogger()
        logger.warning(f'KeyError: Calculation of mesh failed.')
        f.close()
        return None
    except IndexError:
        logger = logging.getLogger()
        logger.warning(f'KeyError: Calculation of mesh failed.')
        f.close()
        return None


if __name__ == '__main__':
    file = Path("/home/lukas/rtm/rtm_files/2019-07-24_16-32-40_308_RESULT.erfh5")
    mesh = get_batched_mesh_dgl(4, file)
    pass