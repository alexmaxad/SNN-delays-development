import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

from DCLS.construct.modules import Dcls1d

from spikingjelly.spikingjelly.activation_based import neuron, layer
from spikingjelly.spikingjelly.activation_based import base



class DaleDcls1d_positive(Dcls1d):
    def __init__(self, *args, **kwargs):
        super(DaleDcls1d_negative, self).__init__(*args, **kwargs)

    def forward(self, input):
        return self._conv_forward(input, torch.abs(self.weight), self.bias, self.P, self.SIG)

class DaleDcls1d_negative(Dcls1d):
    def __init__(self, *args, **kwargs):
        super(DaleDcls1d_negative, self).__init__(*args, **kwargs)

    def forward(self, input):
        return self._conv_forward(input, -torch.abs(self.weight), self.bias, self.P, self.SIG)
    


class DCLS_semi_DANNLayer(nn.Module):

    def __init__(
            self,
            n_inputs,
            n_outputs, 
            kernel_count, 
            groups, 
            dilated_kernel_size, 
            bias, 
            version,
            config,
    ):
        super().__init__()

        self.config = config

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_inputs_exc = round(n_inputs * self.config.exc_proportion)
        self.n_inputs_inh = n_inputs - self.n_inputs_exc

        self.w_exc_exc = nn.Parameter(torch.rand(n_outputs, n_inputs))
        self.w_inh_exc = nn.Parameter(torch.rand(self.n_inputs_inh, n_inputs))
        self.w_exc_inh = nn.Parameter(torch.rand(n_outputs, self.n_inputs_inh))

        if bias:
            self.bias = nn.Parameter(torch.randn(n_outputs))
            fan_in, _ = n_inputs
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

        self.bn_I = layer.BatchNorm1d(self.n_inputs_inh, step_mode='m')
        self.LIF = neuron.LIFNode(tau=self.config.init_tau, v_threshold=self.config.v_threshold, 
                                                       surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                       step_mode='m', decay_input=False, store_v_seq = True)

    '''@property
    def weight(self):
        # Concatenate weights along the second dimension (dim=1)
        return torch.cat((self.w_exc, self.w_inh), dim=1)'''

    def forward(self, x):

        in_inhib = DaleDcls1d_positive(
            in_channels=self.n_inputs,
            out_channels=self.n_inputs_inh,
            kernel_count=self.config.kernel_count,
            groups=1,
            dilated_kernel_size = self.config.max_delay,
            bias=self.config.bias, 
            version=self.config.DCLSversion,
        )
        in_inhib = self.bn_I(in_inhib)
        in_inhib = self.LIF(in_inhib)

        out_inhib = DaleDcls1d_negative(
            in_channels=self.n_inputs_inh,
            out_channels=self.n_outputs,
            kernel_count=self.config.kernel_count,
            groups=1,
            dilated_kernel_size = self.config.max_delay,
            bias=self.config.bias, 
            version=self.config.DCLSversion,
        )

        excit_excit = DaleDcls1d_positive(
            in_channels=self.n_inputs,
            out_channels=self.n_outputs,
            kernel_count=self.config.kernel_count,
            groups=1,
            dilated_kernel_size = self.config.max_delay,
            bias=self.config.bias, 
            version=self.config.DCLSversion,
        )

        out = out_inhib + excit_excit

        if self.bias is not None:
            out += self.bias
            
        return out
