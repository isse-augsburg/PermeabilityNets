import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras, RasterizationSettings, PointLights, \
    MeshRenderer, MeshRasterizer, SoftPhongShader, TexturesVertex
from vedo.mesh import Mesh
from vedo.io import screenshot
from vedo.plotter import show
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
from pathlib import Path
from Utils.mesh_utils import MeshCreator


def show_vedo_mesh_old(verts, faces, filling_factors):
    start = time.time()

    mesh = Mesh([verts, faces])

    mesh.backColor('blue').lineColor('white').lineWidth(0)

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
    show(mesh, __doc__, viewup='z', interactive=False, camera={'pos': (-1, -1, 2)})  # isometric: 2 2 2
    screenshot()
    end = time.time()
    print(f"Calculation took {end - start} seconds.")


class VedoMeshSaver:
    """ First Attempt on parallelizing the texture calculation. Is currently slower than before :( """
    def __init__(self, verts, faces, filling_factors):
        mesh = Mesh([verts, faces])
        self.filling_factors = filling_factors
        self.verts = verts
        self.faces = faces
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

        mesh = Mesh([self.verts, self.faces])
        mesh.backColor('blue').lineColor('white').lineWidth(0)
        mesh.cellIndividualColors(colors)
        show(mesh, __doc__, viewup='z', interactive=False, camera={'pos': (-1, -1, 2)})  # isometric: 2 2 2
        screenshot()
        end = time.time()
        print(f"Calculation took {end - start} seconds.")


def save_p3d_mesh(verts, faces, filling_factors):
    features = [(int(i * 255), 0, 0) for i in filling_factors]
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


if __name__ == '__main__':
    file = Path("/home/lukas/rtm/rtm_files_3d/2020-08-24_11-20-27_111_RESULT.erfh5")
    from Pipeline.data_loader_mesh import DataLoaderMesh
    sensor_verts_path = Path("/home/lukas/rtm/sensor_verts_3d_272_subsampled.dump")
    dl = DataLoaderMesh(sensor_verts_path=sensor_verts_path)
    data = dl.get_sensor_flowfront_mesh(file)
    sample = data[150][1]
    mc = MeshCreator(file)
    verts, faces, _ = mc.get_mesh_components()

    show_vedo_mesh_old(verts, faces, sample)
    # save_p3d_mesh(verts, faces, sample)
    pass
