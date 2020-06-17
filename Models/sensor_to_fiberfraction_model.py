import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from Models.OurModules import ConvLSTM


class STFF_v2(nn.Module):
    def __init__(self, slice_start=0, shrink_factor=1):
        super(STFF_v2, self).__init__()
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
        return out


class STFF(nn.Module):
    def __init__(self):
        super(STFF, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(1140, 1140,
                            batch_first=False, num_layers=2,
                            bidirectional=False, dropout=0)
        self.ff = nn.Linear(1140, 13905)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        lstm_out, hidden = self.lstm(x)

        out = lstm_out[-1]

        out = self.dropout(out)
        out = self.ff(out)
        return out


if __name__ == "__main__":
    model_inpt = torch.randn(8, 100, 1140).cuda()
    model_target = torch.randn(8, 135, 103).cuda()

    model = STFF_v2(shrink_factor=4, slice_start=1).cuda()
    t = model(model_inpt)
    loss = torch.nn.MSELoss()
    lo = loss(t, model_target)
    print(lo)
