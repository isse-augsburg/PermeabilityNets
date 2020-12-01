import torch
from torch import nn
import torch.nn.functional as F
from dgl.nn import GraphConv, TAGConv


class SensorMeshToFlowFrontModelDGL(nn.Module):
    def __init__(self, mesh, batch_size=None):
        super(SensorMeshToFlowFrontModelDGL, self).__init__()

        self.batch_size = batch_size

        if torch.cuda.device_count() > 1:
            self.batch_size = int(self.batch_size / torch.cuda.device_count())
            print(f"Internal batch size: {self.batch_size}")

        self.mesh = mesh

        # This part worked best
        self.gc1 = GraphConv(1, 16)
        self.gc2 = GraphConv(16, 32)
        self.gc3 = GraphConv(32, 32)
        self.gc4 = GraphConv(32, 32)
        self.gc5 = GraphConv(32, 32)
        self.gc6 = GraphConv(32, 32)
        self.gc7 = GraphConv(32, 32)
        self.gc8 = GraphConv(32, 16)
        self.gc9 = GraphConv(16, 16)
        self.gc10 = GraphConv(16, 16)
        self.gc11 = GraphConv(16, 16)
        self.gc12 = GraphConv(16, 8)
        self.gc13 = GraphConv(8, 1)

        # Additional Part for small amount of sensors.
        self.tc1 = TAGConv(1, 8)
        self.tc2 = TAGConv(8, 16)
        self.tc3 = TAGConv(16, 16)
        self.tc4 = TAGConv(16, 1)

        '''self.num_heads = 1
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
        x = F.relu(self.gc5(m, x))

        x = F.relu(self.gc6(m, x))

        x = F.relu(self.gc7(m, x))

        x = F.relu(self.gc8(m, x))

        x = F.relu(self.gc9(m, x))

        x = F.relu(self.gc10(m, x))
        x = F.relu(self.gc11(m, x))
        x = F.relu(self.gc12(m, x))

        x = torch.sigmoid(self.gc13(m, x))
        '''x = F.relu(self.gc13(m, x))

        x = F.relu(self.tc1(m, x))
        x = F.relu(self.tc2(m, x))
        x = F.relu(self.tc3(m, x))
        x = torch.sigmoid(self.tc4(m, x))'''

        x = x.view(self.batch_size, -1)
        return x


class SparseSensorMeshToFlowFrontModelDGL(nn.Module):
    def __init__(self, mesh, batch_size=None):
        super(SparseSensorMeshToFlowFrontModelDGL, self).__init__()

        self.batch_size = batch_size

        if torch.cuda.device_count() > 1:
            self.batch_size = int(self.batch_size / torch.cuda.device_count())
            print(f"Internal batch size: {self.batch_size}")

        self.mesh = mesh

        # This part worked best
        self.tc1 = TAGConv(1, 8, k=5)
        self.tc2 = TAGConv(8, 16, k=3)
        self.gc1 = GraphConv(8, 16)

        self.gc2 = GraphConv(16, 32)
        self.gc3 = GraphConv(32, 32)
        self.gc4 = GraphConv(32, 32)
        self.gc5 = GraphConv(32, 32)
        self.gc6 = GraphConv(32, 32)
        self.gc7 = GraphConv(32, 32)
        self.gc8 = GraphConv(32, 16)
        self.gc9 = GraphConv(16, 16)
        self.gc10 = GraphConv(16, 16)
        self.gc11 = GraphConv(16, 16)
        self.gc12 = GraphConv(16, 8)
        self.gc13 = GraphConv(8, 1)

        # Additional Part for small amount of sensors.
        # self.tc1 = TAGConv(1, 8)
        # self.tc2 = TAGConv(8, 16)
        # self.tc3 = TAGConv(16, 16)
        # self.tc4 = TAGConv(16, 1)

    def forward(self, x):
        m = self.mesh.to(torch.device('cuda:0'))

        x = x.view(-1, 1)

        x = F.relu(self.tc1(m, x))
        x = F.relu(self.tc2(m, x))

        # x = F.relu(self.gc1(m, x))
        # x = x.view(-1, x.size(1) * x.size(2))
        x = F.relu(self.gc2(m, x))
        # x = x.view(-1, x.size(1) * x.size(2))
        x = F.relu(self.gc3(m, x))
        # x = x.view(-1, x.size(1) * x.size(2))
        x = F.relu(self.gc4(m, x))
        # x = x.view(-1, x.size(1) * x.size(2))
        # x = F.relu(self.gc5(m, x))

        # x = F.relu(self.gc6(m, x))

        x = F.relu(self.gc7(m, x))

        x = F.relu(self.gc8(m, x))

        x = F.relu(self.gc9(m, x))

        x = F.relu(self.gc10(m, x))
        x = F.relu(self.gc11(m, x))
        x = F.relu(self.gc12(m, x))

        x = torch.sigmoid(self.gc13(m, x))
        '''x = F.relu(self.gc13(m, x))


        x = F.relu(self.tc2(m, x))
        x = F.relu(self.tc3(m, x))
        x = torch.sigmoid(self.tc4(m, x))'''

        x = x.view(self.batch_size, -1)
        return x
