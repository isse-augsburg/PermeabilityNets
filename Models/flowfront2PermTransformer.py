import socket
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
import math
from Models.utils.decorators import augumented_forward_ff_sequence


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Bumblebee(nn.Module):
    def __init__(self):
        super(Bumblebee, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, stride=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=(0, 1))

        self.pool = nn.MaxPool2d(2, 2)
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, padding=(0, 1), )
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2)
        self.deconv5 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2)
        self.deconv6 = nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((143, 111))

    def forward(self, x):
        out = x.unsqueeze(1)
        out = F.relu(self.pool(self.conv1(out)))
        out = F.relu(self.pool(self.conv2(out)))
        out = F.relu(self.pool(self.conv3(out)))
        out = F.relu(self.pool(self.conv4(out)))
        out = F.relu(self.pool(self.conv5(out)))

        out = F.relu((self.deconv1(out)))
        out = F.relu((self.deconv2(out)))
        out = F.relu((self.deconv3(out)))
        out = F.relu((self.deconv4(out)))
        out = F.relu((self.deconv5(out)))
        out = self.deconv6(out).squeeze(1)
        return self.adaptive_pool(out)


class OptimusPrime(nn.Module):
    def __init__(self, batch_size=32, chpkpt=None):
        super(OptimusPrime, self).__init__()
        self.batch_size = batch_size

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, stride=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=(0, 1))

        self.pool = nn.MaxPool2d(2, 2)
        self.pos_encode = PositionalEncoding(512, max_len=100)
        self.transformer = nn.Transformer(512, 4, 2, 2, 1024)

        self.deconv1 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=5,
            padding=(0, 1),
        )
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=7, stride=2)
        self.deconv5 = nn.ConvTranspose2d(32, 1, kernel_size=7, stride=2)


        self.adaptive_pool = nn.AdaptiveAvgPool2d((135, 103))

        self.trg = torch.ones((1, self.batch_size, 512)).cuda()

        if chpkpt is not None:
            pretrained_dict = torch.load(chpkpt)
            pretrained_dict = pretrained_dict["model_state_dict"]
            model_dict = self.state_dict()

            new_model_state_dict = OrderedDict()
            model_state_dict = pretrained_dict
            if "swt-dgx" not in socket.gethostname():
                for k, v in model_state_dict.items():
                    if k.startswith("module"):
                        k = k[7:]  # remove `module.`
                    new_model_state_dict[k] = v

                model_state_dict = new_model_state_dict

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in model_state_dict.items() if k in model_dict and 'de' not in k}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)

            self.load_state_dict(model_dict)

            for name, param in self.named_parameters():
                if name.startswith('conv'):
                    param.requires_grad = False

    @augumented_forward_ff_sequence
    def forward(self, x: torch.Tensor):
        inpt_batch_size = x.shape[0]
        if inpt_batch_size != self.batch_size:
            trg = torch.ones((1, x.shape[0], 512)).cuda()

        else:
            trg = self.trg

        # create mask for zeros at the end of frame
        padding = torch.zeros((x.shape[0], 100), dtype=torch.bool).cuda()
        for j, sample in enumerate(x):
            for i in range(99, 0, -1):
                step = sample[i]
                n_zero = torch.nonzero(step)
                if len(n_zero) > 0:
                    break
                padding[j, i] = True
            pass

        np_pad = padding.cpu().numpy()
        out = x.view(-1, 1, x.shape[2], x.shape[3])
        out = F.relu(self.pool(self.conv1(out)))
        out = F.relu(self.pool(self.conv2(out)))
        out = F.relu(self.pool(self.conv3(out)))
        out = F.relu(self.pool(self.conv4(out)))
        out = F.relu(self.pool(self.conv5(out))).squeeze(-1).squeeze(-1)
        out = out.view(inpt_batch_size, -1, 512)
        out = out.permute(1, 0, 2)
        out = self.pos_encode(out)
        transformed = self.transformer(out, trg, src_key_padding_mask=padding)
        out = transformed.permute(1, 0, 2).squeeze(1).unsqueeze(-1).unsqueeze(-1)

        out = F.relu(self.deconv1(out))
        out = F.relu(self.deconv2(out))
        out = F.relu(self.deconv3(out))
        out = F.relu(self.deconv4(out))
        out = self.deconv5(out).squeeze(1)

        return out


class OptimusPrime_c2D(nn.Module):
    def __init__(self, batch_size=32, chpkpt=None):
        super(OptimusPrime_c2D, self).__init__()
        self.batch_size = batch_size

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, stride=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=(0, 1))

        self.pool = nn.MaxPool2d(2, 2)
        self.pos_encode = PositionalEncoding(512, max_len=100)
        self.transformer = nn.Transformer(512, 4, 2, 2, 1024)

        self.deconv1 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=5,
            padding=(0, 1),
        )
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=7, stride=2)
        self.deconv5 = nn.ConvTranspose2d(32, 16, kernel_size=7, stride=2)

        self.deconv6 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2)
        self.conv6_1 = nn.Conv2d(16, 1, kernel_size=5, stride=2, padding=1)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((135, 103))

        self.trg = torch.ones((1, self.batch_size, 512)).cuda()

        if chpkpt is not None:
            pretrained_dict = torch.load(chpkpt)
            pretrained_dict = pretrained_dict["model_state_dict"]
            model_dict = self.state_dict()

            new_model_state_dict = OrderedDict()
            model_state_dict = pretrained_dict
            if "swt-dgx" not in socket.gethostname():
                for k, v in model_state_dict.items():
                    if k.startswith("module"):
                        k = k[7:]  # remove `module.`
                    new_model_state_dict[k] = v

                model_state_dict = new_model_state_dict

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in model_state_dict.items() if k in model_dict and 'de' not in k}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)

            self.load_state_dict(model_dict)

            for name, param in self.named_parameters():
                if name.startswith('conv'):
                    param.requires_grad = False

    @augumented_forward_ff_sequence
    def forward(self, x: torch.Tensor):
        inpt_batch_size = x.shape[0]
        if inpt_batch_size != self.batch_size:
            trg = torch.ones((1, x.shape[0], 512)).cuda()

        else:
            trg = self.trg

        # create mask for zeros at the end of frame
        padding = torch.zeros((x.shape[0], 100), dtype=torch.bool).cuda()
        for j, sample in enumerate(x):
            for i in range(99, 0, -1):
                step = sample[i]
                n_zero = torch.nonzero(step)
                if len(n_zero) > 0:
                    break
                padding[j, i] = True
            pass

        np_pad = padding.cpu().numpy()
        out = x.view(-1, 1, x.shape[2], x.shape[3])
        out = F.relu(self.pool(self.conv1(out)))
        out = F.relu(self.pool(self.conv2(out)))
        out = F.relu(self.pool(self.conv3(out)))
        out = F.relu(self.pool(self.conv4(out)))
        out = F.relu(self.pool(self.conv5(out))).squeeze(-1).squeeze(-1)
        out = out.view(inpt_batch_size, -1, 512)
        out = out.permute(1, 0, 2)
        out = self.pos_encode(out)
        transformed = self.transformer(out, trg, src_key_padding_mask=padding)
        out = transformed.permute(1, 0, 2).squeeze(1).unsqueeze(-1).unsqueeze(-1)

        out = F.relu(self.deconv1(out))
        out = F.relu(self.deconv2(out))
        out = F.relu(self.deconv3(out))
        out = F.relu(self.deconv4(out))

        out = F.relu(self.deconv5(out))
        out = F.relu(self.deconv6(out))
        out = self.conv6_1(out).squeeze(1)

        return out


class Bumblebee2(nn.Module):
    def __init__(self):
        super(Bumblebee2, self).__init__()
        self.fc1 = nn.Linear(15873, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 512)

        self.out3 = nn.Linear(4096, 15873)
        self.out2 = nn.Linear(2048, 4096)
        self.out1 = nn.Linear(512, 2048)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor):
        out = torch.flatten(x, start_dim=1)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = F.relu(self.fc3(out))
        out = self.dropout(out)

        out = F.relu(self.out1(out))
        out = self.dropout(out)
        out = F.relu(self.out2(out))
        out = self.dropout(out)
        out = F.relu(self.out3(out))

        out = torch.reshape(out, (-1, 143, 111))

        return out


class OptimusPrime2(nn.Module):
    def __init__(self, batch_size=32, chpkpt=None):
        super(OptimusPrime2, self).__init__()
        self.batch_size = batch_size

        self.fc1 = nn.Linear(15873, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 512)

        self.out3 = nn.Linear(4096, 15873)
        self.out2 = nn.Linear(2048, 4096)
        self.out1 = nn.Linear(512, 2048)

        self.dropout = nn.Dropout(0.2)

        self.pool = nn.MaxPool2d(2, 2)
        self.pos_encode = PositionalEncoding(512, max_len=100)
        self.transformer = nn.Transformer(512, 4, 2, 2, 1024)

        self.deconv1 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=5,
            padding=(0, 1),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((135, 103))

        self.trg = torch.ones((1, self.batch_size, 512)).cuda()

        if chpkpt is not None:
            pretrained_dict = torch.load(chpkpt)
            pretrained_dict = pretrained_dict["model_state_dict"]
            model_dict = self.state_dict()

            new_model_state_dict = OrderedDict()
            model_state_dict = pretrained_dict
            if "swt-dgx" not in socket.gethostname():
                for k, v in model_state_dict.items():
                    if k.startswith("module"):
                        k = k[7:]  # remove `module.`
                    new_model_state_dict[k] = v

                model_state_dict = new_model_state_dict

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in model_state_dict.items() if k in model_dict and 'out' not in k}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)

            self.load_state_dict(model_dict)

            for name, param in self.named_parameters():
                if name.startswith('fc'):
                    param.requires_grad = False

    @augumented_forward_ff_sequence
    def forward(self, x: torch.Tensor):
        inpt_batch_size = x.shape[0]
        if inpt_batch_size != self.batch_size:
            trg = torch.ones((1, x.shape[0], 512)).cuda()

        else:
            trg = self.trg

        # create mask for zeros at the end of frame
        padding = torch.zeros((x.shape[0], 100), dtype=torch.bool).cuda()
        for j, sample in enumerate(x):
            for i in range(99, 0, -1):
                step = sample[i]
                n_zero = torch.nonzero(step)
                if len(n_zero) > 0:
                    break
                padding[j, i] = True
            pass

        out = x.view(-1, 1, x.shape[2], x.shape[3])
        out = torch.flatten(out, start_dim=1)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = F.relu(self.fc3(out))
        out = self.dropout(out)
        out = out.view(inpt_batch_size, -1, 512)
        out = out.permute(1, 0, 2)
        out = self.pos_encode(out)
        transformed = self.transformer(out, trg, src_key_padding_mask=padding)
        out = transformed.permute(1, 0, 2).squeeze(1)

        out = F.relu(self.out1(out))
        out = self.dropout(out)
        out = F.relu(self.out2(out))
        out = self.dropout(out)
        out = F.relu(self.out3(out))

        out = torch.reshape(out, (-1, 143, 111))
        out = self.adaptive_pool(out)

        return out


if __name__ == "__main__":
    model_inpt = torch.randn(2, 100, 143, 111).cuda()
    model_target = torch.randn(2, 135, 103).cuda()

    model = OptimusPrime_c2D(batch_size=2, chpkpt=None).cuda()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    t = model(model_inpt)
    loss = torch.nn.MSELoss()
    lo = loss(t, model_target)
    print(lo)
