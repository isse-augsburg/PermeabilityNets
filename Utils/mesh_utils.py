import h5py
import numpy as np
import torch
import logging
from pytorch3d.structures import Meshes
import dgl
from pathlib import Path
from Utils.data_utils import normalize_coords
from vedo import Mesh, printc, show


def show_vedo_mesh(verts, faces):
    mesh = Mesh([verts, faces])

    mesh.backColor('violet').lineColor('violet').lineWidth(2)
    labs = mesh.labels('id')

    # retrieve them as numpy arrays
    printc('points():\n', mesh.points(), c=3)
    printc('faces(): \n', mesh.faces(), c=3)

    show(mesh, labs, __doc__, viewup='z', axes=1)


class MeshCreator:
    """Class for handling different implementations of triangle meshes.

    Args:
        sample_file (Path): Path to *.erfh5 File from which a mesh should be built.
    """

    def __init__(self, sample_file):
        self.sample_file = sample_file
        self.faces = None,
        self.vertices = None
        self.edges = None

        self.__calculate_mesh_components(self.sample_file)

        self.subsampled_nodes = None

    def __calculate_mesh_components(self, sample_file, normalize_coordinates=True):
        f = h5py.File(sample_file, 'r')

        try:
            verts = f["post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/"
                      "erfblock/res"][()]

            if normalize_coordinates:
                verts = normalize_coords(verts, third_dim=True)

            # Get internal indices of nodes
            hashes = f["post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/"
                       "erfblock/entid"][()]
            hashes = {h: i for i, h in enumerate(hashes)}

            # Calculate faces based on internal indices
            faces = f["post/constant/connectivities/SHELL/erfblock/ic"][()]
            faces = faces[:, :-1]
            faces = np.vectorize(hashes.__getitem__)(faces)

            f.close()
            self.vertices = verts
            self.faces = faces
            self.edges = self.__calculate_edges()

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

    def __calculate_edges(self):
        torch_mesh = self.__torch_mesh()
        return torch_mesh.edges_packed().numpy()

    def batched_mesh_dgl(self, batchsize):
        dgl_mesh = self.__dgl_mesh()
        batch = [dgl_mesh for i in range(batchsize)]
        batched_mesh = dgl.batch(batch)
        return batched_mesh

    def __torch_mesh(self):
        faces = torch.unsqueeze(torch.Tensor(self.faces), dim=0)
        verts = torch.unsqueeze(torch.tensor(self.vertices), dim=0)

        return Meshes(verts=verts, faces=faces)

    def __dgl_mesh(self):
        src, dst = np.split(self.edges, 2, axis=1)
        u = np.squeeze(np.concatenate([src, dst]))
        v = np.squeeze(np.concatenate([dst, src]))
        dgl_mesh = dgl.graph((u, v))
        return dgl_mesh

    def batched_mesh_torch(self, batchsize):
        faces = self.faces.repeat(batchsize, 1, 1)
        verts = self.vertices.repeat(batchsize, 1, 1)
        mesh = Meshes(verts=verts, faces=faces)

        return mesh

    '''def get_open3d_mesh(self, filename):
        verts, faces = self.get_mesh_components(filename)
        verts, faces = np.squeeze(verts.numpy()), np.squeeze(faces.numpy())

        verts = open3d.utility.Vector3dVector(verts)
        faces = open3d.utility.Vector3iVector(faces)

        mesh = open3d.geometry.TriangleMesh(verts, faces)

        return mesh'''

    def subsampled_batched_mesh_dgl(self, batchsize, faces_percentage=0.7):
        self.__subsample_faces(faces_percentage)
        mesh = self.__dgl_mesh()

        batch = [mesh for i in range(batchsize)]
        batched_mesh = dgl.batch(batch)
        batched_mesh = dgl.add_self_loop(batched_mesh)

        return batched_mesh

    def __subsample_faces(self, faces_percentage=0.8):
        idx = np.random.randint(0, len(self.faces), int(len(self.faces) * faces_percentage))
        self.faces = self.faces[idx, :]
        self.edges = self.__calculate_edges()

    def get_subsampled_nodes(self):
        return self.subsampled_nodes

    def get_mesh_components(self):
        return self.vertices, self.faces, self.edges


if __name__ == '__main__':
    file = Path("/home/lukas/rtm/rtm_files_3d/2020-08-24_11-20-27_111_RESULT.erfh5")
    mc = MeshCreator(file)
    m = mc.subsampled_batched_mesh_dgl(4)
    pass
