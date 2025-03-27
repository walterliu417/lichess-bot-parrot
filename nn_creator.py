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
        self.mlp.add_module("Activation 1", nn.LeakyReLU())
        self.mlp.add_module("Layer 2", nn.Linear(500, 250))
        self.mlp.add_module("Activation 2", nn.LeakyReLU())
        self.mlp.add_module("Layer 3", nn.Linear(250, 1))
        self.mlp.add_module("Activation 3", nn.Sigmoid())

    def forward(self, x):
        x = self.conv_net.forward(x)
        return self.mlp.forward(x)
    
class ComplexModel(nn.Module):
  # Attempt at using a deeper CNN.

    def __init__(self, name):
        super().__init__()

        self.name = name

        self.conv_net = nn.Sequential()
        self.conv_net.add_module("Conv 1", nn.Conv2d(1, 64, 2, 1))
        self.conv_net.add_module("Batchnorm 1", nn.BatchNorm2d(64))
        self.conv_net.add_module("Conv activation", nn.LeakyReLU())
        self.conv_net.add_module("Conv 2", nn.Conv2d(64, 87, 2, 1))
        self.conv_net.add_module("Batchnorm 2", nn.BatchNorm2d(87))
        self.conv_net.add_module("Conv activation 2", nn.LeakyReLU())
        self.conv_net.add_module("Conv 3", nn.Conv2d(87, 125, 2, 1))
        self.conv_net.add_module("Batchnorm 3", nn.BatchNorm2d(125))
        self.conv_net.add_module("Conv activation 3", nn.LeakyReLU())
        self.conv_net.add_module("Conv 4", nn.Conv2d(125, 195, 2, 1))
        self.conv_net.add_module("Batchnorm 4", nn.BatchNorm2d(195))
        self.conv_net.add_module("Conv activation 4", nn.LeakyReLU())
        self.conv_net.add_module("Conv 5", nn.Conv2d(195, 347, 2, 1))
        self.conv_net.add_module("Batchnorm 5", nn.BatchNorm2d(347))
        self.conv_net.add_module("Conv activation 5", nn.LeakyReLU())
        self.conv_net.add_module("Conv 6", nn.Conv2d(347, 781, 2, 1))
        self.conv_net.add_module("Batchnorm 6", nn.BatchNorm2d(781))
        self.conv_net.add_module("Conv activation 6", nn.LeakyReLU())
        self.conv_net.add_module("Conv 7", nn.Conv2d(781, 3124, 2, 1))
        self.conv_net.add_module("Batchnorm 7", nn.BatchNorm2d(3124))
        self.conv_net.add_module("Conv activation 7", nn.LeakyReLU())
        self.conv_net.add_module("Flattener", nn.Flatten())

        self.mlp = nn.Sequential()
        self.mlp.add_module("Linear 1", nn.Linear(3136, 1))
        self.mlp.add_module("Activation 1", nn.Sigmoid())


    def forward(self, x, feat):
        x = self.conv_net.forward(x)
        y = torch.column_stack((x, feat))
        return self.mlp.forward(y)

    def count_parameters(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
class MeanModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.mean(a)
