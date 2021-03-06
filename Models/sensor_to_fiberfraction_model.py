from Models.OurModules.AttentionConvLstm import AttenConvLSTM
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from Models.OurModules import ConvLSTM
from Models.utils.decorators import augumented_forward_ff_sequence


class STFF_v2(nn.Module):
    def __init__(self, slice_start=0, shrink_factor=1):
        super(STFF_v2, self).__init__()
        self.slice_start = slice_start
        self.shrink_factor = shrink_factor
        num_addlayers = int(np.log2(self.shrink_factor))
        self.l_add_layers = nn.ModuleList()
        k = 16

        self.convlstm = ConvLSTM(
            input_channels=1,
            hidden_channels=[128, 32],
            kernel_size=3,
            step=100,
            effective_step=[99],
        )
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
        x = x[
            :,
            :,
            :,
            self.slice_start::self.shrink_factor,
            self.slice_start::self.shrink_factor,
        ]
        out, _ = self.convlstm(x)
        out = out[0]
        out = F.relu(self.transpose1(out))
        for layer in self.l_add_layers:
            out = F.relu(layer(out))
        out = self.transpose2(out)
        out = torch.squeeze(out, dim=1)
        out = self.adaptive_pool(out)
        return out


class FFTFF(nn.Module):
    def __init__(self):
        super(FFTFF, self).__init__()

        self.convlstm = ConvLSTM(
            input_channels=1,
            hidden_channels=[32, 32],
            kernel_size=5,
            step=100,
            effective_step=[99],
            dilation=2,
        )
        self.transpose = nn.ConvTranspose2d(32, 8, 5, stride=2, padding=0)
        self.conv1 = nn.Conv2d(8, 8, 13, stride=2, padding=0)
        self.conv2 = nn.Conv2d(8, 1, 5, stride=1, padding=0)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((135, 103))

    @augumented_forward_ff_sequence
    def forward(self, x: torch.Tensor):
        # sequence, batch, dim, x,y
        x = x.permute(1, 0, 2, 3).unsqueeze(2)

        out, _ = self.convlstm(x)
        out = out[0]
        out = F.relu(self.transpose(out))
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        out = torch.squeeze(out, dim=1)
        out = self.adaptive_pool(out)
        return out


class STFF(nn.Module):
    def __init__(self):
        super(STFF, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(
            1140, 1140, batch_first=False, num_layers=2, bidirectional=False, dropout=0
        )
        self.ff = nn.Linear(1140, 13905)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        lstm_out, hidden = self.lstm(x)

        out = lstm_out[-1]

        out = self.dropout(out)
        out = self.ff(out)
        return out


class AttentionFFTFF(nn.Module):
    def __init__(self, method="b"):
        super(AttentionFFTFF, self).__init__()

        # self.convlstm = ConvLSTM(input_channels=1, hidden_channels=[
        #                        32, 32], kernel_size=5, step=100, effective_step=[99], dilation=2)

        self.convlstm = AttenConvLSTM(
            input_channels=1,
            hidden_channels=[32, 32],
            kernel_size=5,
            step=100,
            init_method="xavier_normal_",
            AttenMethod=method,
            expand_x=143,
            expand_y=111,
        )

        self.transpose = nn.ConvTranspose2d(32, 8, 5, stride=2, padding=0)
        self.conv1 = nn.Conv2d(8, 8, 13, stride=2, padding=0)
        self.conv2 = nn.Conv2d(8, 1, 5, stride=1, padding=0)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((135, 103))

    def forward(self, x: torch.Tensor):
        # sequence, batch, dim, x,y
        x = x.unsqueeze(1)

        out, _ = self.convlstm(x)
        out = out[:, :, -1, :, :].squeeze(2)
        out = F.relu(self.transpose(out))
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        out = torch.squeeze(out, dim=1)
        out = self.adaptive_pool(out)
        return out


class ThreeDAttentionFFTFF(nn.Module):
    def __init__(self, method="b"):
        super(ThreeDAttentionFFTFF, self).__init__()

        self.conv3d = nn.Conv3d(1, 16, 5)

        self.convlstm = AttenConvLSTM(
            input_channels=16,
            hidden_channels=[32, 32],
            kernel_size=5,
            step=96,
            init_method="xavier_normal_",
            AttenMethod=method,
            expand_x=139,
            expand_y=107,
        )

        self.transpose = nn.ConvTranspose2d(32, 8, 5, stride=2, padding=0)
        self.conv1 = nn.Conv2d(8, 8, 13, stride=2, padding=0)
        self.conv2 = nn.Conv2d(8, 1, 5, stride=1, padding=0)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((135, 103))

    def forward(self, x: torch.Tensor):
        # sequence, batch, dim, x,y
        x = x.unsqueeze(1)
        out = F.relu(self.conv3d(x))

        out, _ = self.convlstm(out)
        out = out[:, :, -1, :, :].squeeze(2)
        out = F.relu(self.transpose(out))
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        out = torch.squeeze(out, dim=1)
        out = self.adaptive_pool(out)
        return out


if __name__ == "__main__":
    model_inpt = torch.randn(2, 100, 143, 111).cuda()
    model_target = torch.randn(2, 135, 103).cuda()

    model = ThreeDAttentionFFTFF("a").cuda()
    t = model(model_inpt)
    loss = torch.nn.MSELoss()
    lo = loss(t, model_target)
    print(lo)
