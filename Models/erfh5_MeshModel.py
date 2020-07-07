import torch
from torch import nn
import torch.nn.functional as F
from pytorch3d.ops import GraphConv
from tqdm import tqdm
from Models.model_utils import load_model_layers_from_path
import logging


class SensorMeshToFlowFrontModel(nn.Module):
    def __init__(self, mesh, batch_size=None):
        super(SensorMeshToFlowFrontModel, self).__init__()

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


class SensorMeshToDryspotModel(nn.Module):
    def __init__(self, mesh, batch_size=None, bottleneck_dim=1, pretrained='GraphConv', weights_path=None):
        super(SensorMeshToDryspotModel, self).__init__()
        self.mesh = mesh
        self.batch_size = batch_size

        self.bottleneck_dim = bottleneck_dim

        self.gc1 = GraphConv(1, 16)
        self.gc2 = GraphConv(16, 32)
        self.gc3 = GraphConv(32, 64)
        self.gc4 = GraphConv(64, 32)
        self.gc5 = GraphConv(32, bottleneck_dim)

        self.conv1 = nn.Conv2d(bottleneck_dim, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.conv5 = nn.Conv2d(128, 256, 3)

        self.pool = nn.MaxPool2d(2, 2)

        self.avg_pool = nn.AdaptiveAvgPool1d(168*168*bottleneck_dim)
        self.avg_pool2 = nn.AdaptiveAvgPool2d((self.bottleneck_dim, 168, 168))
        self.linear1 = nn.Linear(2304, 1024)
        self.linear2 = nn.Linear(1024, 256)
        self.linear3 = nn.Linear(256, 1)

        if pretrained == 'GraphConv' and weights_path is not None:
            logger = logging.getLogger(__name__)
            weights = load_model_layers_from_path(path=weights_path,
                                                  layer_names={'gc1', 'gc2', 'gc3', 'gc4', 'gc5'})
            incomp = self.load_state_dict(weights, strict=False)
            logger.debug(f'All layers: {self.state_dict().keys()}')
            logger.debug(f'Loaded weights but the following: {incomp}')
            print("Loaded layers")

        #TODO freeze?

        self.count = 0

    def forward(self, x):

        m = self.mesh.cuda()
        edges = m.edges_packed()
        x = x.view(-1, 1).contiguous()

        x = F.relu(self.gc1(x, edges))
        x = F.relu(self.gc2(x, edges))
        x = F.relu(self.gc3(x, edges))
        x = F.relu(self.gc4(x, edges))
        x = F.relu(self.gc5(x, edges))

        x = torch.unsqueeze(x.view(self.batch_size, -1), dim=1)
        x = self.avg_pool(x)
        x = x.view(size=(self.batch_size, self.bottleneck_dim, 168, 168))
        #x = self.avg_pool2(x)

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        # shape: [batch_size, 256, 3, 3]
        x = x.view((x.shape[0], 2304, -1))
        x = x.mean(-1)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))

        return x



if __name__ == '__main__':
    from Pipeline.data_loader_mesh import DataLoaderMesh
    from pathlib import Path
    dl = DataLoaderMesh(sensor_verts_path=Path("/home/lukas/rtm/sensor_verts.dump"))
    file = Path("/home/lukas/rtm/rtm_files/2019-07-24_16-32-40_308_RESULT.erfh5")

    bs = 4

    mesh = dl.get_batched_mesh(bs, file)
    # model = SensorMeshToFlowFrontModel(mesh)
    model = SensorMeshToDryspotModel(mesh, bs).cuda()
    instances = dl.get_sensor_flowfront_mesh(file)
    data, labels = [], []
    batch = instances[0:bs]
    for d, l in batch:
        data.append(d)
        labels.append(l)

    data = torch.Tensor(data).cuda()
    lables = torch.Tensor(labels)

    for i in tqdm(range(500)):
        output = model(data)
