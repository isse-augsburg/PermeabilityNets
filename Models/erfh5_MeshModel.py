
import torch
from torch import nn
import torch.nn.functional as F
from pytorch3d.ops import GraphConv
from tqdm import tqdm


class MeshModel(nn.Module):
    def __init__(self, mesh, input_dim=1, batch_size=None):
        super(MeshModel, self).__init__()

        self.batch_size = batch_size

        self.mesh = mesh

        self.gc1 = GraphConv(1, 16).cuda()
        self.gc2 = GraphConv(16, 32).cuda()
        self.gc3 = GraphConv(32, 64).cuda()
        self.gc4 = GraphConv(64, 32).cuda()
        self.gc5 = GraphConv(32, 1).cuda()

    def forward(self, x):
        m = self.mesh.cuda()
        edges = m.edges_packed()
        x = x.view(-1, 1)

        x = F.relu(self.gc1(x, edges))
        x = F.relu(self.gc2(x, edges))
        x = F.relu(self.gc3(x, edges))
        x = F.relu(self.gc4(x, edges))
        x = torch.sigmoid(self.gc5(x, edges))

        x = x.view(self.batch_size, -1)

        return x


if __name__ == '__main__':
    from Pipeline.data_loader_mesh import DataLoaderMesh
    from pathlib import Path
    dl = DataLoaderMesh(sensor_verts_path=Path("/home/lukas/rtm/sensor_verts.dump"))
    file = Path("/home/lukas/rtm/rtm_files/2019-07-24_16-32-40_308_RESULT.erfh5")

    bs = 96

    mesh = dl.get_batched_mesh(bs, file)
    model = MeshModel(mesh)
    instances = dl.get_sensor_flowfront_mesh(file)
    data, labels = [], []
    batch = instances[0:bs]
    for d, l in batch:
        data.append(d)
        labels.append(l)

    data = torch.Tensor(data).cuda()
    lables = torch.Tensor(labels)

    for i in tqdm(range(500)):
        model(data)
