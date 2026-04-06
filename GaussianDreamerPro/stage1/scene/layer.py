import torch
import torch.nn as nn
import torch.nn.functional as F

def make_linear_layers(feat_dims, relu_final=True, use_gn=False):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i+1]))

        # Do not use ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and relu_final):
            if use_gn:
                layers.append(nn.GroupNorm(4, feat_dims[i+1]))
            #layers.append(nn.ReLU(inplace=True))
            layers.append(nn.GELU())

    return nn.Sequential(*layers)

# subtract sigmoid output by 0.5 -> 0 value if x=0
class Sigmoid_0origin(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(x) - 0.5
