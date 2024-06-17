import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DANNLayer(nn.Module):
    def __init__(self, in_features, out_features, exc_proportion, bias, step_mode):
        super().__init__()

        self.in_features = in_features
        
        self.in_features_exc = round(out_features * exc_proportion)
        self.in_features_inh = in_features - self.in_features_exc

        # Initialize parameters
        self.WEE = nn.Parameter(torch.empty(out_features, self.in_features_exc))
        self.WIE = nn.Parameter(torch.empty(out_features, self.in_features_exc))
        self.WEI = nn.Parameter(torch.empty(self.in_features_exc, self.in_features_inh))
        self.alpha = nn.Parameter(torch.empty(self.in_features_inh, 1))
        self.g = nn.Parameter(torch.empty(self.in_features_exc, 1))
        self.beta = nn.Parameter(torch.empty(self.in_features_exc, 1))

        self.step_mode = step_mode

        self.init_weights()

    def init_weights(self):
        lambda_E = math.sqrt(self.in_features * (2 * math.pi - 1) / (2 * math.pi))
        exponential_dist = torch.distributions.exponential.Exponential(lambda_E)

        with torch.no_grad():
            self.WEE.copy_(exponential_dist.sample(self.WEE.shape))
        
            if self.in_features_inh == 1:
                self.WIE.fill_((1 / self.in_features_exc) * torch.sum(self.WEE, dim=1))
                self.WEI.fill_(1)
            else:
                self.WIE.copy_(exponential_dist.sample(self.WIE.shape))
                self.WEI.fill_(1 / self.in_features_exc)
        
        self.alpha.fill_(math.log(math.sqrt(2 * math.pi - 1) / math.sqrt(self.in_features)))
        self.g.fill_(1)
        self.beta.fill_(1)

    def forward(self, x):
        # x is expected to be of shape (time, batch_size, ne)
        
        zE = torch.mm(self.WEE, x.T) #(ne, batch_size)
        hI = torch.mm(self.WIE, x.T) #(ni, batch_size)
        
        inhibition_effect = torch.mm(self.WEI, hI) #(ne, batch_size)
        
        gamma = torch.mm(self.WEI,  hI * torch.exp(self.alpha)) + 1e-8 #(ne, batch_size)

        z = (self.g/gamma) * (zE - inhibition_effect) + self.beta.T #(ne, batch_size)
        
        h = F.relu(z) #(ne, batch_size)
        
        return h

        

    def update_weights(self, eta, grads):
        with torch.no_grad():
            self.WEE -= eta * grads['WEE']
            self.WIE -= eta * grads['WIE']
            self.WEI -= eta * grads['WEI']
            self.alpha -= eta * grads['alpha']
            self.g -= eta * grads['g']
            self.beta -= eta * grads['beta']
            
            self.WEE.data = torch.clamp(self.WEE.data, min=0)
            self.WIE.data = torch.clamp(self.WIE.data, min=0)
            self.WEI.data = torch.clamp(self.WEI.data, min=0)
            self.g.data = torch.clamp(self.g.data, min=0)

    def correct_gradients(self, grads):
        grads['WEE'] = grads['WEE']
        grads['WIE'] = grads['WIE'] / math.sqrt(self.ne)
        grads['WEI'] = grads['WEI'] / self.in_features
        grads['alpha'] = grads['alpha'] / (self.in_features * math.sqrt(self.ne))
        grads['g'] = grads['g']
        grads['beta'] = grads['beta']
        return grads