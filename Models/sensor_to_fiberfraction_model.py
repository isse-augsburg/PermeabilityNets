import torch
from torch import nn
import torch.nn.functional as F
from Models.OurModules import ConvLSTM


class STFF_v2(nn.Module):
    def __init__(self):
        super(STFF_v2, self).__init__()
        self.convlstm = ConvLSTM(input_channels=1, hidden_channels=[
                                 128, 32], kernel_size=3, step=100, effective_step=[99])
        self.transpose1 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=0)
        self.transpose2 = nn.ConvTranspose2d(16, 1, 5, stride=2, padding=0)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((135, 103))

    def forward(self, x: torch.Tensor):
        # sequence, batch, dim, x,y
        x = x.permute(1, 0, 2).reshape((100, -1, 1, 38, 30))
        out, _ = self.convlstm(x)
        out = out[0]
        out = F.relu(self.transpose1(out))
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

    model = STFF_v2().cuda()
    t = model(model_inpt)
    loss = torch.nn.MSELoss()
    lo = loss(t, model_target)
    print(lo)
