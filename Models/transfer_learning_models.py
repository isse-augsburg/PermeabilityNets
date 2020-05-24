import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                      bias=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, 1)

    def forward(self, x):
        out = self.model(x)
        out = F.sigmoid(out)
        return out