import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

from DCLS.construct.modules import Dcls1d

from spikingjelly.spikingjelly.activation_based import neuron, layer
from spikingjelly.spikingjelly.activation_based import base



class DaleDcls1d_positive(Dcls1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            bias, 
            config,
    ):
        super().__init__()

        self.config = config

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_inputs_inh = round(n_outputs * self.config.inh_proportion)

        self.w_exc_inh = nn.Parameter(torch.rand(self.n_outputs, self.n_inputs_inh))

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
        
        self.DCLS_inh = DaleDcls1d_positive(
            in_channels=self.n_inputs,
            out_channels=self.n_inputs_inh,
            kernel_count=self.config.kernel_count,
            groups=1,
            dilated_kernel_size = self.config.max_delay,
            bias=self.config.bias, 
            version=self.config.DCLSversion,
        )

        self.DCLS_exc = DaleDcls1d_positive(
            in_channels=self.n_inputs,
            out_channels=self.n_outputs,
            kernel_count=self.config.kernel_count,
            groups=1,
            dilated_kernel_size = self.config.max_delay,
            bias=self.config.bias, 
            version=self.config.DCLSversion,
        )

        self.DCLS_layers = [self.DCLS_inh, self.DCLS_exc]

        '''# Adding a DCLS after intermediate inhibitory layer to try :
        self.DCLS_inter = DaleDcls1d_negative(
            in_channels=self.n_inputs_inh,
            out_channels=self.n_outputs,
            kernel_count=self.config.kernel_count,
            groups=1,
            dilated_kernel_size = self.config.max_delay,
            bias=self.config.bias, 
            version=self.config.DCLSversion,
        )
'''

    def forward(self, x):

        # x is of size (batch, neurons, time)

        #print(f'x_size = {x.size()}')

        if self.n_inputs_inh > 0:
            in_inhib = self.DCLS_inh(x)
            in_inhib = in_inhib.permute(2,0,1) # (time, batch, neurons)
            in_inhib = self.bn_I(in_inhib)
            in_inhib = self.LIF(in_inhib)
        
            #print(f'w_exc_inh = {self.w_exc_inh.size()}')

            out_inhib = F.linear(in_inhib, -torch.abs(self.w_exc_inh))
            out_inhib = out_inhib.permute(1,2,0) # (batch, neurons, time)

            # Trying DCLS after intermediate layer :
            #in_inhib = in_inhib.permute(1,2,0) # (batch, neurons, time)
            #in_inhib = F.pad(in_inhib, (self.config.left_padding, 0), 'constant', 0)

            #print(f'in_inihib after padding = {in_inhib.size()}') #, {self.bn_I}')

            #out_inhib = self.DCLS_inter(in_inhib)

        #print(f'out_inhib = {out_inhib.size()}')

        excit_excit = self.DCLS_exc(x)

        #print(f'excit_excit = {excit_excit.size()}')

        out = excit_excit + out_inhib

        #print(f'out = {out.size()}')

        if self.bias is not None:
            out += self.bias
            
        return out
