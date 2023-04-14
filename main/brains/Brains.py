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
        #self.net = nn.Sequential(nn.Linear(n_input,n_output),nn.ReLU6())
        self.net = nn.Sequential(nn.Linear(n_input,n_output),nn.Tanh())
        ##self.apply(self._init_weights)

        if parentnet==None:
            self.init_brainA1()
        else:
            with torch.no_grad():
                self.net[0].weight =  nn.Parameter( parentnet[0].weight +torch.tensor(np.random.normal(loc=0.0, scale=.01, size=(3,3))), requires_grad=False)
                self.net[0].bias =   nn.Parameter( parentnet[0].bias +torch.tensor(np.random.normal(loc=0.0, scale=.01, size=(3))), requires_grad=False)

    def init_brainA1(self):
        ii = np.random.randint(1,3)
        if ii == 1:
            with torch.no_grad():
                self.net[0].weight = nn.Parameter(torch.tensor(np.array(
                    [[ 0.2195,  1.8272,  0.1061],
                     [ 0.6854,  0.7245, -0.3855],
                     [ 0.6534,  0.3376,  0.1650]]
                    )  + np.random.normal(loc=0.0, scale=.01, size=(3,3))
                    ), requires_grad=False)
                self.net[0].bias = nn.Parameter(torch.tensor(
                    np.array([-0.1455,  0.0630,  0.1198]) + np.random.normal(loc=0.0, scale=.01, size=3)
                    ), requires_grad=False)
        elif ii==2:
            with torch.no_grad():
                self.net[0].weight = nn.Parameter(torch.tensor(np.array(
                    [[ 0.1974,  1.7478,  0.0948],
                     [ 0.9434,  0.6146, -0.4172],
                     [ 0.5559,  0.4153,  0.3030]]
                    )  + np.random.normal(loc=0.0, scale=.01, size=(3,3))
                    ), requires_grad=False)
                self.net[0].bias = nn.Parameter(torch.tensor(
                    np.array([-0.0996, -0.0344, -0.0675]) + np.random.normal(loc=0.0, scale=.01, size=3)
                    ), requires_grad=False)
        elif ii==3:
            with torch.no_grad():
                self.net[0].weight = nn.Parameter(torch.tensor(np.array(
                    [[ 0.4287,  1.8374,  0.1616],
                     [ 0.9340,  0.5540, -0.3759],
                     [ 0.6628,  0.3212,  0.0666]]
                    )  + np.random.normal(loc=0.0, scale=.01, size=(3,3))
                    ), requires_grad=False)
                self.net[0].bias = nn.Parameter(torch.tensor(
                    np.array([0.0639, 0.0916, 0.1126]) + np.random.normal(loc=0.0, scale=.01, size=3)
                    ), requires_grad=False)

#    def _init_weights(self, module):
#        if isinstance(module, nn.Linear):
#            module.weight.data.normal_(mean=0.0, std=1.0)
#            if module.bias is not None:
#                module.bias.data.zero_()

    def _freeze_param(self):
        for k,v in self.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.net(x)
            return x

    def printbrain(self):
        print(self.net)
        for name, param in self.net.named_parameters():
            print(name, param.shape,param.requires_grad)
            print(param)

    def printbrain_to_file(self,file1):
            print(self.net,file=file1)
            for name, param in self.net.named_parameters():
                print(name, param.shape,param.requires_grad,file=file1)
                print(param,file=file1)

    def reproduce(self):
        newbrain = annBrain(3,3, self.net )
        return newbrain 




class rnnBrain(nn.Module):

    # Number of features used as input. (Number of columns)
    INPUT_SIZE = 3
    # Number of previous time stamps taken into account.
    SEQ_LENGTH = 1
    # Number of features in last hidden state ie. number of output time-
    # steps to predict.See image below for more clarity.
    HIDDEN_SIZE = 6
    # Number of stacked rnn layers.
    NUM_LAYERS = 1
    BATCH_SIZE = 1

    def __init__(self,n_input,n_output,parentnet=None):
        torch.set_default_dtype(torch.float64)
        super().__init__()
        self.n_input=n_input
        self.n_output=n_output
        self._freeze_param()
        #self.net = nn.Sequential(nn.Linear(n_input,n_output),nn.Sigmoid())

        self.net = nn.RNN(input_size=n_input, hidden_size=self.HIDDEN_SIZE, num_layers = self.NUM_LAYERS, batch_first=True)
        self._freeze_param()

        if parentnet==None:
            self.init_brainA1()
        else:
            with torch.no_grad():
                self.net.weight_ih_l0 =   nn.Parameter( parentnet.weight_ih_l0 +torch.tensor(np.random.normal(loc=0.0, scale=.01, size=(self.HIDDEN_SIZE,3))), requires_grad=False)
                self.net.weight_hh_l0 =   nn.Parameter( parentnet.weight_hh_l0 +torch.tensor(np.random.normal(loc=0.0, scale=.01, size=(self.HIDDEN_SIZE,self.HIDDEN_SIZE))), requires_grad=False)
                self.net.bias_ih_l0 =   nn.Parameter( parentnet.bias_ih_l0 +torch.tensor(np.random.normal(loc=0.0, scale=.01, size=(self.HIDDEN_SIZE))), requires_grad=False)
                self.net.bias_ih_l0 =   nn.Parameter( parentnet.bias_ih_l0 +torch.tensor(np.random.normal(loc=0.0, scale=.01, size=(self.HIDDEN_SIZE))), requires_grad=False)

    def init_brainA1(self):
        ii = np.random.randint(1,3)
#        if ii == 1:
        with torch.no_grad():
                self.net.weight_ih_l0 = nn.Parameter(torch.tensor(np.array(
                    [[-0.0706, -0.1586,  0.1262],
                    [-0.1394, -0.0662, -0.0818],
                    [-0.0753,  0.0357, -0.1296],
                    [ 0.2903,  1.1314,  0.1728],
                    [ 0.6691,  0.0520,  0.1075],
                    [ 0.1995,  0.1139,  0.7112]],
                    )  + np.random.normal(loc=0.0, scale=.1, size=(self.HIDDEN_SIZE,3))
                    ), requires_grad=False)
                self.net.weight_hh_l0 = nn.Parameter(torch.tensor(np.array(
                    [[-0.0954,  0.1125, -0.1212, -0.0522, -0.0308, -0.1376],
                    [-0.1538, -0.0662, -0.2202, -0.0414,  0.1605, -0.0763],
                    [ 0.1592,  0.0115, -0.1116, -0.0135, -0.2496,  0.1311],
                    [ 0.2584, -0.1853, -0.0563,  0.0859,  0.1063, -0.1611],
                    [-0.2139, -0.0630,  0.0538, -0.0242, -0.1810,  0.1133],
                    [-0.1655,  0.1978,  0.1442,  0.0524,  0.1080,  0.0099]]
                    ) + 0.01*np.identity(self.HIDDEN_SIZE) + np.random.normal(loc=0.0, scale=.001, size=(self.HIDDEN_SIZE,self.HIDDEN_SIZE))
                    ), requires_grad=False)
                self.net.bias_ih_l0 = nn.Parameter(torch.tensor(
                     np.array([-0.1707,  0.2708,  0.1886,  0.1015,  0.0957,  0.0282]) + np.random.normal(loc=0.0, scale=.001, size=(self.HIDDEN_SIZE))
                    ), requires_grad=False)
                self.net.bias_hh_l0 = nn.Parameter(torch.tensor(
                     np.array([0.2261, 0.1568, 0.2925, 0.3509, 0.1058, 0.1845]) + np.random.normal(loc=0.0, scale=.001, size=(self.HIDDEN_SIZE))
                    ), requires_grad=False)

    def _freeze_param(self):
        for k,v in self.named_parameters():
            v.requires_grad = False

    def forward(self,x):
        with torch.no_grad():
            #print(x.reshape(1,1,len(x)))
            out, h_n = self.net( x.reshape(1,1,len(x))  )
            return out[0,0,-3:]

    def printbrain(self):
            print(self.net)
            for name, param in self.net.named_parameters():
                print(name, param.shape,param.requires_grad)
                print(param)

    def printbrain_to_file(self,file1):
            print(self.net,file=file1)
            for name, param in self.net.named_parameters():
                print(name, param.shape,param.requires_grad,file=file1)
                print(param,file=file1)

    def reproduce(self):
        newbrain = rnnBrain(3,3, self.net )
        return newbrain 



