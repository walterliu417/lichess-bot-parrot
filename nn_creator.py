import torch
import torch.nn as nn

class UltraSimpleModel(nn.Module):

    def __init__(self, name):
        super().__init__()

        self.name = name

        self.conv_net = nn.Sequential()
        self.conv_net.add_module("Conv 1", nn.Conv2d(1, 64, 8, 1))
        self.conv_net.add_module("Conv batchnorm 1", nn.BatchNorm2d(64, momentum=0.2))
        self.conv_net.add_module("Conv activation 1", nn.LeakyReLU())
        self.conv_net.add_module("Flattener", nn.Flatten())

        self.mlp = nn.Sequential()
        self.mlp.add_module("Layer 1", nn.Linear(76, 1))
        self.mlp.add_module("Activation 1", nn.Sigmoid())

    def forward(self, x, feat):
        x = self.conv_net.forward(x)
        y = torch.column_stack((x, feat))
        return self.mlp.forward(y)

    def count_parameters(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)

class SimpleModel(nn.Module):

    def __init__(self, name):
        super().__init__()

        self.name = name

        self.conv_net = nn.Sequential()
        self.conv_net.add_module("Conv 1", nn.Conv2d(1, 100, 2, 1))
        self.conv_net.add_module("Batchnorm 1", nn.BatchNorm2d(100))
        self.conv_net.add_module("Conv activation", nn.LeakyReLU())
        self.conv_net.add_module("Conv 2", nn.Conv2d(100, 250, 2, 1))
        self.conv_net.add_module("Batchnorm 2", nn.BatchNorm2d(250))
        self.conv_net.add_module("Conv activation 2", nn.LeakyReLU())
        self.conv_net.add_module("Conv 3", nn.Conv2d(250, 400, 3, 1))
        self.conv_net.add_module("Batchnorm 3", nn.BatchNorm2d(400))
        self.conv_net.add_module("Conv activation 3", nn.LeakyReLU())
        self.conv_net.add_module("Conv 4", nn.Conv2d(400, 700, 4, 1))
        self.conv_net.add_module("Batchnorm 4", nn.BatchNorm2d(700))
        self.conv_net.add_module("Conv activation 4", nn.LeakyReLU())
        self.conv_net.add_module("Flattener", nn.Flatten())

        self.mlp = nn.Sequential()
        self.mlp.add_module("Layer 1", nn.Linear(700, 500))
        self.mlp.add_module("Batchnorm 1", nn.BatchNorm1d(500))
        self.mlp.add_module("Activation 1", nn.LeakyReLU())
        self.mlp.add_module("Layer 2", nn.Linear(500, 250))
        self.mlp.add_module("Activation 2", nn.LeakyReLU())
        self.mlp.add_module("Layer 3", nn.Linear(250, 1))
        self.mlp.add_module("Activation 3", nn.Sigmoid())

    def forward(self, x):
        x = self.conv_net.forward(x)
        return self.mlp.forward(x)

    def count_parameters(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

class ComplexModel(nn.Module):
  # Attempt at using a deeper CNN.

    def __init__(self, name):
        super().__init__()

        self.name = name

        self.conv_net = nn.Sequential()
        self.conv_net.add_module("Conv 1", nn.Conv2d(1, 64, 3, 1, 1))
        self.conv_net.add_module("SE 1", SE_Block(64))
        self.conv_net.add_module("Batchnorm 1", nn.BatchNorm2d(64))
        self.conv_net.add_module("Conv activation", nn.LeakyReLU())
        self.conv_net.add_module("Conv 2", nn.Conv2d(64, 128, 3, 1, 1))
        self.conv_net.add_module("SE 2", SE_Block(128))
        self.conv_net.add_module("Batchnorm 2", nn.BatchNorm2d(128))
        self.conv_net.add_module("Conv activation 2", nn.LeakyReLU())
        self.conv_net.add_module("Conv 3", nn.Conv2d(128, 192, 3, 1, 1))
        self.conv_net.add_module("SE 3", SE_Block(192))
        self.conv_net.add_module("Batchnorm 3", nn.BatchNorm2d(192))
        self.conv_net.add_module("Conv activation 3", nn.LeakyReLU())
        self.conv_net.add_module("Conv 4", nn.Conv2d(192, 256, 3, 1, 1))
        self.conv_net.add_module("SE 4", SE_Block(256))
        self.conv_net.add_module("Batchnorm 4", nn.BatchNorm2d(256))
        self.conv_net.add_module("Conv activation 4", nn.LeakyReLU())
        self.conv_net.add_module("Conv 5", nn.Conv2d(256, 192, 3, 1, 1))
        self.conv_net.add_module("SE 5", SE_Block(192))
        self.conv_net.add_module("Batchnorm 5", nn.BatchNorm2d(192))
        self.conv_net.add_module("Conv activation 5", nn.LeakyReLU())
        self.conv_net.add_module("Conv 6", nn.Conv2d(192, 128, 3, 1, 1))
        self.conv_net.add_module("SE 6", SE_Block(128))
        self.conv_net.add_module("Batchnorm 6", nn.BatchNorm2d(128))
        self.conv_net.add_module("Conv activation 6", nn.LeakyReLU())
        self.conv_net.add_module("Conv 7", nn.Conv2d(128, 64, 3, 1, 1))
        self.conv_net.add_module("SE 7", SE_Block(64))
        self.conv_net.add_module("Batchnorm 7", nn.BatchNorm2d(64))
        self.conv_net.add_module("Conv activation 7", nn.LeakyReLU())
        self.conv_net.add_module("Flattener", nn.Flatten())

        self.mlp = nn.Sequential()
        self.mlp.add_module("Linear 1", nn.Linear(4096, 1))
        self.mlp.add_module("Activation 1", nn.Sigmoid())

    def forward(self, x):
        a=self.conv_net(x)
        return self.mlp(self.conv_net(x))

    def count_parameters(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)



class MeanModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.mean(a)

