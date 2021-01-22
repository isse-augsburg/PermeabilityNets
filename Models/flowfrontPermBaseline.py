import torch
import torch.nn as nn
import torch.nn.functional as F
class FF2Perm_Baseline(nn.Module):
    def __init__(self,):
        super(FF2Perm_Baseline, self).__init__()
        

        self.conv1 = nn.Conv2d(100, 64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, stride=1,padding=2)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=5, stride=1,padding=2)
        self.conv4 = nn.Conv2d(16, 1, kernel_size=5, stride=1)
       


        self.adaptive_pool = nn.AdaptiveAvgPool2d((135, 103))

      
    def forward(self, x: torch.Tensor):
        out = x
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = self.adaptive_pool(out).squeeze(1)

        

        return out




if __name__ == "__main__":
    model_inpt = torch.randn(2, 100, 143, 111).cuda()
    model_target = torch.randn(2, 135, 103).cuda()

    model = FF2Perm_Baseline().cuda()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    t = model(model_inpt)
    loss = torch.nn.MSELoss()
    lo = loss(t, model_target)
    print(lo)
