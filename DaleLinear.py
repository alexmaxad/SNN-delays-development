import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from spikingjelly.spikingjelly.activation_based import base

class DaleLinear(nn.Module, base.StepModule):

    def __init__(self, in_features, out_features, exc_proportion, bias, step_mode):
        super().__init__()

        self.in_features_exc = round(in_features * exc_proportion)
        self.in_features_inh = in_features - self.in_features_exc

        self.w_exc = nn.Parameter(torch.rand(out_features, self.in_features_exc))
        self.w_inh = nn.Parameter(torch.rand(out_features, self.in_features_inh))

        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

        self.step_mode = step_mode

    @property
    def weight(self):
        # Concatenate weights along the second dimension (dim=1)
        return torch.cat((self.w_exc, self.w_inh), dim=1)

    def forward(self, x):

        x_exc, x_inh = x[:, :, :self.in_features_exc], x[:, :, self.in_features_exc:]

        out = 0

        if self.in_features_exc > 0 :
            out_exc = F.linear(x_exc, torch.abs(self.w_exc))
            out += out_exc
        if self.in_features_inh > 0 :
            out_inh = F.linear(x_inh, -torch.abs(self.w_inh))
            out += out_inh

        if self.bias is not None:
            out += self.bias
            
        return out
