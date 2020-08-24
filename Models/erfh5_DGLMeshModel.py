import torch
from torch import nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from tqdm import tqdm
from Models.model_utils import load_GraphConv_layers_from_path
import torchvision.models as m
import logging

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

    def forward(self, x):
        m = self.mesh.to(torch.device('cuda:0'))

        x = x.view(-1, 1)

        x = F.relu(self.gc1(m, x))
        x = F.relu(self.gc2(m, x))
        x = F.relu(self.gc3(m, x))
        x = F.relu(self.gc4(m, x))
        x = torch.sigmoid(self.gc5(m, x))

        x = x.view(self.batch_size, -1)

        return x