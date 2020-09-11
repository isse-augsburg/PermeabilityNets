import torch
from torch import nn
import torch.nn.functional as F
from dgl.nn import GraphConv


class SensorMeshToFlowFrontModelDGL(nn.Module):
    def __init__(self, mesh, batch_size=None):
        super(SensorMeshToFlowFrontModelDGL, self).__init__()

        self.batch_size = batch_size

        self.mesh = mesh

        self.gc1 = GraphConv(1, 16)
        self.gc2 = GraphConv(16, 32)
        self.gc3 = GraphConv(32, 64)
        self.gc4 = GraphConv(64, 32)
        self.gc5 = GraphConv(32, 1)

        '''self.num_heads = 3
        self.gc1 = GATConv(1, 8, num_heads=self.num_heads)
        self.gc2 = GATConv(8*self.num_heads, 16, num_heads=self.num_heads)
        self.gc3 = GATConv(16*self.num_heads, 32, num_heads=self.num_heads)
        self.gc4 = GATConv(32*self.num_heads, 16, num_heads=self.num_heads)
        self.gc5 = GATConv(16*self.num_heads, 1, num_heads=1)'''

        '''self.gc1 = TAGConv(1, 8)
        self.gc2 = TAGConv(8, 16)
        self.gc3 = TAGConv(16, 32)
        self.gc4 = TAGConv(32, 16)
        self.gc5 = TAGConv(16, 1)'''

    def forward(self, x):
        m = self.mesh.to(torch.device('cuda:0'))

        x = x.view(-1, 1)

        x = F.relu(self.gc1(m, x))
        # x = x.view(-1, x.size(1) * x.size(2))
        x = F.relu(self.gc2(m, x))
        # x = x.view(-1, x.size(1) * x.size(2))
        x = F.relu(self.gc3(m, x))
        # x = x.view(-1, x.size(1) * x.size(2))
        x = F.relu(self.gc4(m, x))
        # x = x.view(-1, x.size(1) * x.size(2))
        x = torch.sigmoid(self.gc5(m, x))

        x = x.view(self.batch_size, -1)
        return x
