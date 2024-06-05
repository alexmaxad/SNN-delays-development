import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.spikingjelly.activation_based import base

class DaleLinear(nn.Module, base.StepModule):

    def __init__(self, in_features, out_features, exc_proportion, bias, step_mode):
        super().__init__()

        self.in_features_exc = round(in_features * exc_proportion)
        self.in_features_inh = in_features - self.in_features_exc

        self.w_exc = nn.Parameter(torch.rand(out_features, self.in_features_exc))
        self.w_inh = nn.Parameter(torch.rand(out_features, self.in_features_inh))

        self.weight = torch.cat((self.w_exc, self.w_inh), 1)

        if bias == True :
            self.bias = nn.Parameter(torch.rand(out_features))
        else :
            self.bias = nn.Parameter(torch.zeros(out_features))

        self.step_mode = step_mode

    def forward(self, x):

        x_exc, x_inh = x[:, :, :self.in_features_exc], x[:, :, self.in_features_exc:]

        out_exc = F.linear(x_exc, torch.abs(self.w_exc))

        if self.in_features_inh > 0 :
            out_inh = F.linear(x_inh, torch.abs(self.w_inh))
            out = out_exc + out_inh + self.bias
        else :
            out = out_exc + self.bias

        return out
