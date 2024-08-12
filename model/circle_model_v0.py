import torch
from torch import nn


class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=5)
        self.layer2 = nn.Linear(in_features=5, out_features=10)
        self.layer3 = nn.Linear(in_features=10, out_features=20)
        self.layer4 = nn.Linear(in_features=20, out_features=10)
        self.layer5 = nn.Linear(in_features=10, out_features=5)
        self.layer6 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        return self.layer6(self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x))))))