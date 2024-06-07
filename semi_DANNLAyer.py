import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from spikingjelly.spikingjelly.activation_based import neuron, layer
from spikingjelly.spikingjelly.activation_based import base

class semi_DANNLayer(nn.Module, base.StepModule):

    def __init__(self, in_features, out_features, exc_proportion, bias, step_mode, config):
        super().__init__()

        self.config = config

        self.in_features_exc = round(in_features * exc_proportion)
        self.in_features_inh = in_features - self.in_features_exc

        self.w_exc_exc = nn.Parameter(torch.rand(out_features, in_features))
        self.w_inh_exc = nn.Parameter(torch.rand(self.in_features_inh, in_features))
        self.w_exc_inh = nn.Parameter(torch.rand(out_features, self.in_features_inh))

        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

        self.bn_I = layer.BatchNorm1d(self.in_features_inh, step_mode='m')
        self.LIF = neuron.LIFNode(tau=self.config.init_tau, v_threshold=self.config.v_threshold, 
                                                       surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                       step_mode='m', decay_input=False, store_v_seq = True)

        self.step_mode = step_mode

    '''@property
    def weight(self):
        # Concatenate weights along the second dimension (dim=1)
        return torch.cat((self.w_exc, self.w_inh), dim=1)'''

    def forward(self, x):

        '''in_inhib = F.linear(x, torch.abs(self.w_inh_exc))
        in_inhib = self.bn_I(in_inhib)
        in_inhib = self.LIF(in_inhib)

        out_inhib = F.linear(in_inhib, -torch.abs(self.w_exc_inh))'''
        excit_excit = F.linear(x, torch.abs(self.w_exc_exc))

        '''out = out_inhib + excit_excit'''
        out = excit_excit

        if self.bias is not None:
            out += self.bias
            
        return out
