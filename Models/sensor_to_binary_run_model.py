import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from Models.OurModules import ConvLSTM


class SensorToBinaryRunwiseModel(nn.Module):
    def __init__(self, input_dim=1140, slice_start=0, shrink_factor=1):
        super(SensorToBinaryRunwiseModel, self).__init__()
        self.input_dim = input_dim
        self.slice_start = slice_start
        self.shrink_factor = shrink_factor

        num_addlayers = int(np.log2(self.shrink_factor))
        self.l_add_layers = nn.ModuleList()
        k = 16

        self.convlstm = ConvLSTM(input_channels=1, hidden_channels=[
                                 128, 32], kernel_size=3, step=100, effective_step=[99])
        self.transpose1 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=0)

        for _ in range(num_addlayers):
            r = k // 2
            self.l_add_layers.append(nn.ConvTranspose2d(k, r, 3, stride=2, padding=0))
            k = r

        self.transpose2 = nn.ConvTranspose2d(k, 1, 5, stride=2, padding=0)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((135, 103))
        self.fc1 = nn.Linear(135 * 103, 2048)
        self.fc2 = nn.Linear(2048, 1)

    def forward(self, x: torch.Tensor):
        # sequence, batch, dim, x,y
        x = x.permute(1, 0, 2).reshape((100, -1, 1, 38, 30))
        x = x[:, :, :, self.slice_start::self.shrink_factor, self.slice_start::self.shrink_factor]
        out, _ = self.convlstm(x)
        out = out[0]
        out = F.relu(self.transpose1(out))
        for layer in self.l_add_layers:
            out = F.relu(layer(out))
        out = self.transpose2(out)
        out = torch.squeeze(out, dim=1)
        out = self.adaptive_pool(out)
        torch.flatten(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
