import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu6


class annBrain(nn.Module):

    def __init__(self,n_input,n_output,parentnet=None):
        torch.set_default_dtype(torch.float64)
        super().__init__()
        self.n_input=n_input
        self.n_output=n_output
        self._freeze_param()
        #self.net = nn.Sequential(nn.Linear(n_input,n_output),nn.Sigmoid())
        self.net = nn.Sequential(nn.Linear(n_input,n_output),nn.ReLU6())
        self.apply(self._init_weights)

        if parentnet==None:
            with torch.no_grad():
                self.net[0].weight = nn.Parameter(torch.tensor(np.array(
                    [[ .1, 1., .1],
                    [ .5, .1, 0.],
                    [0.,  0.1, .5]])  + np.random.normal(loc=0.0, scale=.5, size=(3,3))
                    ), requires_grad=False)
                self.net[0].bias = nn.Parameter(torch.tensor(np.array(
                    [ 0.1, 0., 0.]) + np.random.normal(loc=0.0, scale=.02, size=3)
                    ), requires_grad=False)
        else:
            with torch.no_grad():
                self.net[0].weight =  nn.Parameter( parentnet[0].weight +torch.tensor(np.random.normal(loc=0.0, scale=.1, size=(3,3))), requires_grad=False)
                self.net[0].bias =   nn.Parameter( parentnet[0].bias +torch.tensor(np.random.normal(loc=0.0, scale=.01, size=(3))), requires_grad=False)
#        for name, param in self.net.named_parameters():
#            print(name, param.shape,param.requires_grad)
#            print(param)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def _freeze_param(self):
        for k,v in self.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.net(x)
            return x

    def print_ann(self):
        print(self.net)
        for name, param in self.net.named_parameters():
            print(name, param.shape,param.requires_grad)
            print(param)

    def reproduce(self):
        newbrain = annBrain(3,3, self.net )
        return newbrain 




