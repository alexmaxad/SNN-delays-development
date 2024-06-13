import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.spikingjelly.activation_based import neuron, layer
from spikingjelly.spikingjelly.activation_based import functional

from DaleDCLSLayer import DCLS_semi_DANNLayer

from model import Model
from utils import set_seed


class SnnDelays_Dale(Model):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
    
    # Try factoring this method
    # Check ThresholdDependent batchnorm (in spikingjelly)
    def build_model(self):

        ########################### Model Description :
        #
        #  self.blocks = (n_layers,  0:weights+bn  |  1: lif+dropout+(synapseFilter) ,  element in sub-block)
        #


        ################################################   First Layer    #######################################################

        self.blocks = [[[DCLS_semi_DANNLayer(self.config.n_inputs, self.config.n_hidden_neurons, bias=self.config.bias, config=self.config)],
                        [layer.Dropout(self.config.dropout_p, step_mode='m')]]]
        
        if self.config.use_batchnorm: self.blocks[0][0].insert(1, layer.BatchNorm1d(self.config.n_hidden_neurons, step_mode='m'))
        if self.config.spiking_neuron_type == 'lif': 
            self.blocks[0][1].insert(0, neuron.LIFNode(tau=self.config.init_tau, v_threshold=self.config.v_threshold, 
                                                       surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                       step_mode='m', decay_input=False, store_v_seq = True))

        elif self.config.spiking_neuron_type == 'plif': 
            self.blocks[0][1].insert(0, neuron.ParametricLIFNode(init_tau=self.config.init_tau, v_threshold=self.config.v_threshold, 
                                                       surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                       step_mode='m', decay_input=False, store_v_seq = True))
        
        elif self.config.spiking_neuron_type == 'heaviside': 
            self.blocks[0][1].insert(0, self.config.surrogate_function)


        if self.config.stateful_synapse:
            self.blocks[0][1].append(layer.SynapseFilter(tau=self.config.stateful_synapse_tau, learnable=self.config.stateful_synapse_learnable, 
                                                         step_mode='m'))


        ################################################   Hidden Layers    #######################################################

        for i in range(self.config.n_hidden_layers-1):
            self.block = [[DCLS_semi_DANNLayer(self.config.n_hidden_neurons, self.config.n_hidden_neurons, bias=self.config.bias, config=self.config)],
                        [layer.Dropout(self.config.dropout_p, step_mode='m')]]
        
            if self.config.use_batchnorm: self.block[0].insert(1, layer.BatchNorm1d(self.config.n_hidden_neurons, step_mode='m'))
            if self.config.spiking_neuron_type == 'lif': 
                self.block[1].insert(0, neuron.LIFNode(tau=self.config.init_tau, v_threshold=self.config.v_threshold, 
                                                       surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                       step_mode='m', decay_input=False, store_v_seq = True))
            elif self.config.spiking_neuron_type == 'plif': 
                self.block[1].insert(0, neuron.ParametricLIFNode(init_tau=self.config.init_tau, v_threshold=self.config.v_threshold, 
                                                       surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                       step_mode='m', decay_input=False, store_v_seq = True))
            
            elif self.config.spiking_neuron_type == 'heaviside': 
                self.block[1].insert(0, self.config.surrogate_function)
            
            if self.config.stateful_synapse:
                self.block[1].append(layer.SynapseFilter(tau=self.config.stateful_synapse_tau, learnable=self.config.stateful_synapse_learnable, 
                                                             step_mode='m'))

            self.blocks.append(self.block)


        ################################################   Final Layer    #######################################################


        self.final_block = [[DCLS_semi_DANNLayer(self.config.n_hidden_neurons, self.config.n_outputs, bias=self.config.bias, config=self.config)]]
        if self.config.spiking_neuron_type == 'lif':
            self.final_block.append([neuron.LIFNode(tau=self.config.init_tau, v_threshold=self.config.output_v_threshold, 
                                                    surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                    step_mode='m', decay_input=False, store_v_seq = True)])
        elif self.config.spiking_neuron_type == 'plif': 
            self.final_block.append([neuron.ParametricLIFNode(init_tau=self.config.init_tau, v_threshold=self.config.output_v_threshold, 
                                                    surrogate_function=self.config.surrogate_function, detach_reset=self.config.detach_reset, 
                                                    step_mode='m', decay_input=False, store_v_seq = True)])



        self.blocks.append(self.final_block)

        self.model = [l for block in self.blocks for sub_block in block for l in sub_block]
        self.model = nn.Sequential(*self.model)
        #print(self.model)

        self.positions_exc_exc = []
        self.positions_inh_exc = []
        self.weights_exc_exc = []
        self.weights_inh_exc = []
        self.weights_exc_inh = []
        self.weights_bn = []
        self.weights_plif = []

        # Trying intermediate DCLS :
        #self.positions_inter = []
        #self.weights_inter = []


        for m in self.model.modules():
            if isinstance(m, DCLS_semi_DANNLayer):
                self.positions_exc_exc.append(m.DCLS_exc.P)
                self.positions_inh_exc.append(m.DCLS_inh.P)
                self.weights_exc_exc.append(m.DCLS_exc.weight)
                self.weights_inh_exc.append(m.DCLS_inh.weight)
                self.weights_exc_inh.append(m.w_exc_inh)

                # Trying intermediate DCLS :
                #self.positions_inter.append(m.DCLS_inter.P)
                #self.weights_inter.append(m.DCLS_inter.weight)

                if self.config.bias:
                    self.weights_bn.append(m.bias)
            elif isinstance(m, layer.BatchNorm1d):
                self.weights_bn.append(m.weight)
                self.weights_bn.append(m.bias)
            elif isinstance(m, neuron.ParametricLIFNode):
                self.weights_plif.append(m.w)



    def init_model(self):

        set_seed(self.config.seed)
        self.mask = []

        if self.config.init_w_method == 'kaiming_uniform':
            for i in range(self.config.n_hidden_layers+1):
                # can you replace with self.weights ?
                torch.nn.init.kaiming_uniform_(self.blocks[i][0][0].DCLS_inh.weight, nonlinearity='relu')
                torch.nn.init.kaiming_uniform_(self.blocks[i][0][0].DCLS_exc.weight, nonlinearity='relu')
                torch.nn.init.kaiming_uniform_(self.blocks[i][0][0].w_exc_inh, nonlinearity='relu')

                # Intermediate DCLS :
                #torch.nn.init.kaiming_uniform_(self.blocks[i][0][0].DCLS_inter.weight, nonlinearity='relu')
                
                '''if self.config.sparsity_p > 0:
                    with torch.no_grad():
                        self.mask.append(torch.rand(self.blocks[i][0][0].weight.size()).to(self.blocks[i][0][0].weight.device))
                        self.mask[i][self.mask[i]>self.config.sparsity_p]=1
                        self.mask[i][self.mask[i]<=self.config.sparsity_p]=0
                        #self.blocks[i][0][0].weight = torch.nn.Parameter(self.blocks[i][0][0].weight * self.mask[i])
                        self.blocks[i][0][0].weight *= self.mask[i]'''


        if self.config.init_pos_method == 'uniform':
            for i in range(self.config.n_hidden_layers+1):
                # can you replace with self.positions?
                torch.nn.init.uniform_(self.blocks[i][0][0].DCLS_inh.P, a = self.config.init_pos_a, b = self.config.init_pos_b)
                torch.nn.init.uniform_(self.blocks[i][0][0].DCLS_exc.P, a = self.config.init_pos_a, b = self.config.init_pos_b)
                self.blocks[i][0][0].DCLS_inh.clamp_parameters()
                self.blocks[i][0][0].DCLS_exc.clamp_parameters()

                # Intermediate DCLS :
                #torch.nn.init.uniform_(self.blocks[i][0][0].DCLS_inter.P, a = self.config.init_pos_a, b = self.config.init_pos_b)
                #self.blocks[i][0][0].DCLS_inter.clamp_parameters()

                if self.config.model_type == 'snn_delays_lr0':
                    self.blocks[i][0][0].DCLS_inh.P.requires_grad = False
                    self.blocks[i][0][0].DCLS_exc.P.requires_grad = False

        for i in range(self.config.n_hidden_layers+1):
            # can you replace with self.positions?
            torch.nn.init.constant_(self.blocks[i][0][0].DCLS_inh.SIG, self.config.sigInit)
            torch.nn.init.constant_(self.blocks[i][0][0].DCLS_exc.SIG, self.config.sigInit)
            self.blocks[i][0][0].DCLS_inh.SIG.requires_grad = False
            self.blocks[i][0][0].DCLS_exc.SIG.requires_grad = False

            # Intermediate DCLS :
            #torch.nn.init.constant_(self.blocks[i][0][0].DCLS_inter.SIG, self.config.sigInit)
            #self.blocks[i][0][0].DCLS_inter.SIG.requires_grad = False



    def reset_model(self, train=True):
        functional.reset_net(self)

        '''for i in range(self.config.n_hidden_layers+1):                
            if self.config.sparsity_p > 0:
                with torch.no_grad():
                    self.mask[i] = self.mask[i].to(self.blocks[i][0][0].weight.device)
                    #self.blocks[i][0][0].weight = torch.nn.Parameter(self.blocks[i][0][0].weight * self.mask[i])
                    self.blocks[i][0][0].weight *= self.mask[i]'''

        # We use clamp_parameters of the Dcls1d modules
        if train: 
            for block in self.blocks:
                block[0][0].DCLS_inh.clamp_parameters()
                block[0][0].DCLS_exc.clamp_parameters()

                # Intermediate DCLS 
                #block[0][0].DCLS_inter.clamp_parameters()




    def decrease_sig(self, epoch):

        # Decreasing to 0.23 instead of 0.5

        alpha = 0
        sigs  = [self.blocks[-1][0][0].DCLS_exc.SIG[0,0,0,0].detach().cpu().item(), self.blocks[-1][0][0].DCLS_inh.SIG[0,0,0,0].detach().cpu().item()] #, self.blocks[-1][0][0].DCLS_inter.SIG[0,0,0,0].detach().cpu().item()]
        if self.config.decrease_sig_method == 'exp':

            if epoch < self.config.final_epoch and sigs[0] > 0.23:
                if self.config.DCLSversion == 'max':
                    # You have to change this !!
                    alpha = (1/self.config.sigInit)**(1/(self.config.final_epoch))
                elif self.config.DCLSversion == 'gauss':
                    alpha = (0.23/self.config.sigInit)**(1/(self.config.final_epoch))

                for block in self.blocks:
                    block[0][0].DCLS_exc.SIG *= alpha
                    # No need to clamp after modifying sigma
                    #block[0][0].clamp_parameters()

            if epoch < self.config.final_epoch and sigs[1] > 0.23:
                if self.config.DCLSversion == 'max':
                    # You have to change this !!
                    alpha = (1/self.config.sigInit)**(1/(self.config.final_epoch))
                elif self.config.DCLSversion == 'gauss':
                    alpha = (0.23/self.config.sigInit)**(1/(self.config.final_epoch))

                for block in self.blocks:
                    block[0][0].DCLS_inh.SIG *= alpha
                    # No need to clamp after modifying sigma
                    #block[0][0].clamp_parameters()

            '''# Intermediate DCLS :
            if epoch < self.config.final_epoch and sigs[2] > 0.23:
                if self.config.DCLSversion == 'max':
                    # You have to change this !!
                    alpha = (1/self.config.sigInit)**(1/(self.config.final_epoch))
                elif self.config.DCLSversion == 'gauss':
                    alpha = (0.23/self.config.sigInit)**(1/(self.config.final_epoch))

                for block in self.blocks:
                    block[0][0].DCLS_inter.SIG *= alpha
                    # No need to clamp after modifying sigma
                    #block[0][0].clamp_parameters()'''





    def forward(self, x):
        
        for block_id in range(self.config.n_hidden_layers):
            # x is permuted: (time, batch, neurons) => (batch, neurons, time)  in order to be processed by the convolution

            x = x.permute(1,2,0)
            #print(f'x size after first permutation = {x.size()}')
            x = F.pad(x, (self.config.left_padding, self.config.right_padding), 'constant', 0)  # we use padding for the delays kernel

            # we use convolution of delay kernels
            x = self.blocks[block_id][0][0](x)
            #print(f'x size after conv block = {x.size()}')

            # We permute again: (batch, neurons, time) => (time, batch, neurons) in order to be processed by batchnorm or Lif
            x = x.permute(2,0,1)

            if self.config.use_batchnorm:
                # we use x.unsqueeze(3) to respect the expected shape to batchnorm which is (time, batch, channels, length)
                # we do batch norm on the channels since length is the time dimension
                # we use squeeze to get rid of the channels dimension 
                x = self.blocks[block_id][0][1](x.unsqueeze(3)).squeeze()
                #print(f'x size after bn = {x.size()}')
            
            # we use our spiking neuron filter
            if self.config.spiking_neuron_type != 'heaviside':
                spikes = self.blocks[block_id][1][0](x)
            else:
                spikes = self.blocks[block_id][1][0](x - self.config.v_threshold)
            # we use dropout on generated spikes tensor


            x = self.blocks[block_id][1][1](spikes)
            #print(f'x size after spikes = {x.size()}')

            # we apply synapse filter
            if self.config.stateful_synapse:
                x = self.blocks[block_id][1][2](x)
            
            # x is back to shape (time, batch, neurons)
        
        # Finally, we apply same transforms for the output layer
        x = x.permute(1,2,0)
        x = F.pad(x, (self.config.left_padding, self.config.right_padding), 'constant', 0)
        
        # Apply final layer
        out = self.blocks[-1][0][0](x)

        # permute out: (batch, neurons, time) => (time, batch, neurons)  For final spiking neuron filter
        out = out.permute(2,0,1)

        if self.config.spiking_neuron_type != 'heaviside':
            out = self.blocks[-1][1][0](out)

            if self.config.loss != 'spike_count':
                out = self.blocks[-1][1][0].v_seq
        
        #print(f'out size after spikes = {out.size()}')

        return out#, self.blocks[0][1][0].v_seq
    


    def get_model_wandb_logs(self):


        sig_exc = self.blocks[-1][0][0].DCLS_exc.SIG[0,0,0,0].detach().cpu().item()

        model_logs = {"sigma":sig_exc}

        for i in range(len(self.blocks)):
            
            '''if self.config.spiking_neuron_type != 'heaviside':
                tau_m = self.blocks[i][1][0].tau if self.config.spiking_neuron_type == 'lif' else  1. / self.blocks[i][1][0].w.sigmoid()
            else: tau_m = 0
            
            if self.config.stateful_synapse and i<len(self.blocks)-1:
                tau_s = self.blocks[i][1][2].tau if not self.config.stateful_synapse_learnable else  1. / self.blocks[i][1][2].w.sigmoid()
            else: tau_s = 0'''
            
            w_m_exc_exc = torch.abs(self.blocks[i][0][0].DCLS_exc.weight).mean()
            w_m_inh_exc = torch.abs(self.blocks[i][0][0].DCLS_inh.weight).mean()
            w_m_exc_inh = torch.abs(self.blocks[i][0][0].w_exc_inh).mean()

            model_logs.update({f'w_exc_exc_{i}':w_m_exc_exc,
                               f'w_inh_exc_{i}':w_m_inh_exc,
                               f'w_exc_inh_{i}':w_m_exc_inh})

        return model_logs


    def round_pos(self):
        with torch.no_grad():
            for i in range(len(self.blocks)):
                self.blocks[i][0][0].DCLS_inh.P.round_()
                self.blocks[i][0][0].DCLS_inh.clamp_parameters()
                self.blocks[i][0][0].DCLS_exc.P.round_()
                self.blocks[i][0][0].DCLS_exc.clamp_parameters()

                # Intermediate DCLS :
                #self.blocks[i][0][0].DCLS_inter.P.round_()
                #self.blocks[i][0][0].DCLS_inter.clamp_parameters()

