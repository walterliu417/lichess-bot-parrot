import torch
import torch.nn as nn

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
        self.mlp.add_module("Layer 1", nn.Linear(712, 500))
        self.mlp.add_module("Activation 1", nn.LeakyReLU())
        self.mlp.add_module("Layer 2", nn.Linear(500, 250))
        self.mlp.add_module("Activation 2", nn.LeakyReLU())
        self.mlp.add_module("Layer 3", nn.Linear(250, 1))
        self.mlp.add_module("Activation 3", nn.Sigmoid())

    def forward(self, x, feat):
        x = self.conv_net.forward(x)
        y = torch.column_stack((x, feat))
        return self.mlp.forward(y)

