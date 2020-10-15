import h5py
import numpy as np
import torch
import logging
from pytorch3d.structures import Meshes
from pytorch3d.renderer import *
import dgl
from pathlib import Path
from Utils.data_utils import normalize_coords
from vedo import *
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool


def show_vedo_mesh_old(verts, faces, filling_factors):
    start = time.time()

    mesh = Mesh([verts, faces])

    mesh.backColor('blue').lineColor('white').lineWidth(0)
    labs = mesh.labels('id')

    # retrieve them as numpy arrays
    # printc('points():\n', mesh.points(), c=3)
    # printc('faces(): \n', mesh.faces(), c=3)

    # show(mesh, labs, __doc__, viewup='z', axes=1)

    colors = []
    all_cells = mesh.faces()
    for i in range(mesh.NCells()):
        points = all_cells[i]
        ff_sum = 0
        for p in points:
            ff_sum += filling_factors[p]

        c = int((ff_sum / 3) * 200)
        colors.append((c, 0, 0))

    mesh.cellIndividualColors(colors)
    show(mesh, __doc__, viewup='z', interactive=False, camera={'pos':(-1,-1,2)}) # isometric: 2 2 2
    screenshot()
    end = time.time()
    print(f"Calculation took {end - start} seconds.")


class VedoMeshSaver:
    def __init__(self, verts, faces, filling_factors):
        mesh = Mesh([verts, faces])
        self.filling_factors = filling_factors

        self.all_cells = mesh.faces()
        self.n_cells = mesh.NCells()

    def calc_color(self, cell):

        points = self.all_cells[cell]
        ff_sum = 0
        for p in points:
            ff_sum += self.filling_factors[p]

        c = int((ff_sum / 3) * 200)
        return (c, 0, 0)

    def show_vedo_mesh(self):
        start = time.time()


        # retrieve them as numpy arrays
        # printc('points():\n', mesh.points(), c=3)
        # printc('faces(): \n', mesh.faces(), c=3)

        # show(mesh, labs, __doc__, viewup='z', axes=1)


        with Pool(processes=8) as pool:
            colors = pool.map(self.calc_color, range(self.n_cells))

        mesh = Mesh([verts, faces])
        mesh.backColor('blue').lineColor('white').lineWidth(0)
        labs = mesh.labels('id')
        mesh.cellIndividualColors(colors)
        show(mesh, __doc__, viewup='z', interactive=False, camera={'pos':(-1,-1,2)}) # isometric: 2 2 2
        screenshot()
        end = time.time()
        print(f"Calculation took {end - start} seconds.")



def save_p3d_mesh(verts, faces, filling_factors):
    features = [(int(i*255), 0, 0) for i in filling_factors]
    features = torch.unsqueeze(torch.Tensor(features), 0)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    texture = TexturesVertex(features)
    mesh = Meshes(torch.unsqueeze(torch.Tensor(verts), 0), torch.unsqueeze(torch.Tensor(faces), 0), texture).cuda()


    # Initialize a camera.
    # Rotate the object by increasing the elevation and azimuth angles
    R, T = look_at_view_transform(dist=2.0, elev=-50, azim=-90)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
    # the difference between naive and coarse-to-fine rasterization.
    raster_settings = RasterizationSettings(
        image_size=1024,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Place a point light in front of the object. As mentioned above, the front of the cow is facing the
    # -z direction.
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    # Create a phong renderer by composing a rasterizer and a shader. The textured phong shader will
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and
    # apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )

    img = renderer(mesh)
    plt.figure(figsize=(10, 10))
    plt.imshow(img[0].cpu().numpy())
    plt.show()

    pass



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
    from Pipeline.data_loader_mesh import DataLoaderMesh
    sensor_verts_path = Path("/home/lukas/rtm/sensor_verts_3d_272_subsampled.dump")
    dl = DataLoaderMesh(sensor_verts_path=sensor_verts_path)
    data = dl.get_sensor_flowfront_mesh(file)
    sample = data[150][1]
    mc = MeshCreator(file)
    # m = mc.subsampled_batched_mesh_dgl(4)
    verts, faces, _ = mc.get_mesh_components()

    show_vedo_mesh_old(verts, faces, sample)
    # save_p3d_mesh(verts, faces, sample)
    pass
