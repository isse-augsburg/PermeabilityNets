
import torch
from torch import nn
import torch.nn.functional as F


class OptimusPrime(nn.Module):
    def __init__(self, batch_size = 32):
        super(OptimusPrime, self).__init__()
        self.batch_size = batch_size

        self.conv1 = nn.Conv2d(1,32, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(32,64, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(64,128, kernel_size=5, stride=1)
        self.conv4 = nn.Conv2d(128,256, kernel_size=5, stride=1)
        self.conv5 = nn.Conv2d(256,512, kernel_size=3, stride=1, padding=(0,1))
        
        self.pool = nn.MaxPool2d(2,2)
        self.transformer = nn.Transformer(512,4,2,2,1024)

        self.deconv1 = nn.ConvTranspose2d(512,256, kernel_size=5, padding=(0,1), )
        self.deconv2 = nn.ConvTranspose2d(256,128, kernel_size=5, stride=2 )
        self.deconv3 = nn.ConvTranspose2d(128,64, kernel_size=6, stride=2 )
        self.deconv4 = nn.ConvTranspose2d(64,32, kernel_size=7, stride=2 )
        self.deconv5 = nn.ConvTranspose2d(32,1, kernel_size=7, stride=2 )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((135, 103))

        self.trg = torch.ones((1,self.batch_size,512)).cuda()

    def forward(self, x: torch.Tensor):
        inpt_batch_size = x.shape[0]
        if(inpt_batch_size != self.batch_size):
            trg = torch.ones((1,x.shape[0],512)).cuda()
        else:
            trg = self.trg
        out = x.view(-1, 1,  x.shape[2], x.shape[3])
        out = F.relu(self.pool(self.conv1(out)))
        out = F.relu(self.pool(self.conv2(out)))
        out = F.relu(self.pool(self.conv3(out)))
        out = F.relu(self.pool(self.conv4(out)))
        out = F.relu(self.pool(self.conv5(out))).squeeze(-1).squeeze(-1)
        out = out.view(inpt_batch_size, -1, 512)
        out = out.permute(1,0,2)
        
        transformed = self.transformer(out,trg)
        out = transformed.permute(1,0,2).squeeze(1).unsqueeze(-1).unsqueeze(-1)

        out = F.relu(self.deconv1(out))
        out = F.relu(self.deconv2(out))
        out = F.relu(self.deconv3(out))
        out = F.relu(self.deconv4(out))
        out = self.deconv5(out).squeeze(1)



        return out


if __name__ == "__main__":
    model_inpt = torch.randn(2, 100, 143, 111).cuda()
    model_target = torch.randn(2, 135, 103).cuda()

    model = OptimusPrime(batch_size=2).cuda()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    t = model(model_inpt)
    loss = torch.nn.MSELoss()
    lo = loss(t, model_target)
    print(lo)
