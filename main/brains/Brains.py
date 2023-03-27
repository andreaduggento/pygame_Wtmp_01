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
        self.net = nn.Sequential(nn.Linear(n_input,n_output),nn.Tanh())
        ##self.apply(self._init_weights)

        if parentnet==None:
            with torch.no_grad():
                self.net[0].weight = nn.Parameter(torch.tensor(np.array(
                    [[ .1, 1., .1],
                    [ .5, .1, 0.],
                    [0.,  0.1, .5]])  + np.random.normal(loc=0.0, scale=.5, size=(3,3))
                    ), requires_grad=False)
                self.net[0].bias = nn.Parameter(torch.tensor(
                    0.01*np.ones(3) + np.random.normal(loc=0.0, scale=.001, size=3)
                    ), requires_grad=False)
        else:
            with torch.no_grad():
                self.net[0].weight =  nn.Parameter( parentnet[0].weight +torch.tensor(np.random.normal(loc=0.0, scale=.01, size=(3,3))), requires_grad=False)
                self.net[0].bias =   nn.Parameter( parentnet[0].bias +torch.tensor(np.random.normal(loc=0.0, scale=.01, size=(3))), requires_grad=False)

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

        ### # input size : (batch, seq_len, input_size)
        ### inputs = data.view(BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
        ### # out shape = (batch, seq_len, num_directions * hidden_size)
        ### # h_n shape  = (num_layers * num_directions, batch, hidden_size)
        ### out, h_n = rnn(inputs)

        if parentnet==None:
            with torch.no_grad():
                self.net.weight_ih_l0 = nn.Parameter(torch.tensor(np.array(
                    [[ 0., 0.,  0.],
                    [ 0., 0.,  0.],
                    [ 0., 0.,  0.],
                    [ .1, 1., .1 ],
                    [ .5, .1, 0. ],
                    [0.,  0.1, .5]],
                    )  + np.random.normal(loc=0.0, scale=.1, size=(self.HIDDEN_SIZE,3))
                    ), requires_grad=False)
                self.net.weight_hh_l0 = nn.Parameter(torch.tensor(np.array(
                    [[ 0., 0., 0., 0., 0., 0. ],
                    [  0., 0., 0., 0., 0., 0. ],
                    [  0., 0., 0., 0., 0., 0. ],
                    [  .1, 0., 0., 0., 0., 0. ],
                    [  0., .1, 0., 0., 0., 0. ],
                    [  0., 0., .1, 0., 0., 0. ]]
                    ) + 0.01*np.identity(self.HIDDEN_SIZE) + np.random.normal(loc=0.0, scale=.001, size=(self.HIDDEN_SIZE,self.HIDDEN_SIZE))
                    ), requires_grad=False)
                self.net.bias_ih_l0 = nn.Parameter(torch.tensor(
                    0.01*np.ones(self.HIDDEN_SIZE) + np.random.normal(loc=0.0, scale=.001, size=(self.HIDDEN_SIZE))
                    ), requires_grad=False)
                self.net.bias_hh_l0 = nn.Parameter(torch.tensor(
                     0.0001* np.ones(self.HIDDEN_SIZE) + np.random.normal(loc=0.0, scale=.001, size=(self.HIDDEN_SIZE))
                    ), requires_grad=False)
        else:
            with torch.no_grad():
                self.net.weight_ih_l0 =   nn.Parameter( parentnet.weight_ih_l0 +torch.tensor(np.random.normal(loc=0.0, scale=.01, size=(self.HIDDEN_SIZE,3))), requires_grad=False)
                self.net.weight_hh_l0 =   nn.Parameter( parentnet.weight_hh_l0 +torch.tensor(np.random.normal(loc=0.0, scale=.01, size=(self.HIDDEN_SIZE,self.HIDDEN_SIZE))), requires_grad=False)
                self.net.bias_ih_l0 =   nn.Parameter( parentnet.bias_ih_l0 +torch.tensor(np.random.normal(loc=0.0, scale=.01, size=(self.HIDDEN_SIZE))), requires_grad=False)
                self.net.bias_ih_l0 =   nn.Parameter( parentnet.bias_ih_l0 +torch.tensor(np.random.normal(loc=0.0, scale=.01, size=(self.HIDDEN_SIZE))), requires_grad=False)

        self.printbrain()

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

    def reproduce(self):
        newbrain = rnnBrain(3,3, self.net )
        return newbrain 



