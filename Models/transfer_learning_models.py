import torch
import torch.nn as nn


class ModelWrapper(nn.Module):
    """
    Wrapper for pretrained torchvision models. Changes the last layer.
    """
    def __init__(self, model, out_features=1):
        super(ModelWrapper, self).__init__()
        self.model = model
        '''self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                      bias=False)'''
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, out_features)

    def forward(self, x):
        out = self.model(x)
        out = torch.sigmoid(out)
        return out
