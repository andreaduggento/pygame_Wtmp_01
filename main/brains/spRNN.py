import torch
from torch import nn
import numpy as np
from nn.Module import RNNBase

class sprnnBrain(nn.Module):
    # Number of features used as input. (Number of columns)
    INPUT_SIZE = 3
    # Number of previous time stamps taken into account.
    SEQ_LENGTH = 1
    # Number of features in last hidden state ie. number of output time-
    # steps to predict.See image below for more clarity.
    HIDDEN_SIZE = 6
    OUTPUT_SIZE = 3
    # Number of stacked rnn layers.
    NUM_LAYERS = 1
    BATCH_SIZE = 1

    self.Apre
    self.Apost

    self.apre = np.zeros(HIDDEN,SIZE)
    self.apost = np.zeros(HIDDEN,SIZE)

    def __init__(self,n_input,n_output,parentnet=None):
        super(sprnnBrain,self).__init__()

        self.K = INPUT_SIZE
        self.N = HIDDEN_SIZE
        self.L = OUTPUT_SIZE
        self.P = n_rnn_layers
        self.rnn = nn.RNN(
                input_size  = self.K,
                hidden_size = self.N,
                num_layers  = self.P,
                nonlinearity = 'relu',
                batch_first = True
                )
        self._freeze_param()

        self.net = nn.Sequential(nn.Linear(self.N,self.L),nn.Tanh())

    def forward(self,x):
        with torch.no_grad():
            #print(x.reshape(1,1,len(x)))
            out, h_n = self.rnn( x.reshape(1,1,len(x))  )
            out = self.net(out)
            return out[0,0]



    if parentnet==None:
        return
#            self.init_brainA1()
        else:
            with torch.no_grad():
                self.net[0].weight =  nn.Parameter( parentnet[0].weight +torch.tensor(np.random.normal(loc=0.0, scale=.01, size=(self.N,self.L))), requires_grad=False)
                self.net[0].bias =   nn.Parameter( parentnet[0].bias +torch.tensor(np.random.normal(loc=0.0, scale=.01, size=(self.L))), requires_grad=False)

            with torch.no_grad():
                self.net.weight_ih_l0 =   nn.Parameter( parentnet.weight_ih_l0 +torch.tensor(np.random.normal(loc=0.0, scale=.01, size=(self.HIDDEN_SIZE,3))), requires_grad=False)
                self.net.weight_hh_l0 =   nn.Parameter( parentnet.weight_hh_l0 +torch.tensor(np.random.normal(loc=0.0, scale=.01, size=(self.HIDDEN_SIZE,self.HIDDEN_SIZE))), requires_grad=False)
                self.net.bias_ih_l0 =   nn.Parameter( parentnet.bias_ih_l0 +torch.tensor(np.random.normal(loc=0.0, scale=.01, size=(self.HIDDEN_SIZE))), requires_grad=False)
                self.net.bias_ih_l0 =   nn.Parameter( parentnet.bias_ih_l0 +torch.tensor(np.random.normal(loc=0.0, scale=.01, size=(self.HIDDEN_SIZE))), requires_grad=False)






    def _freeze_param(self):
        for k,v in self.rnn.named_parameters():
            v.requires_grad = False



    def forward(self, x, hidden):
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)
        batch_size = x.size(0)
        
        # get RNN outputs
        r_out, hidden = self.rnn(x, hidden)
        # shape output to be (batch_size*seq_length, hidden_dim)
        r_out = r_out.view(-1, self.hidden_dim)  
        
        # get final output 
        output = self.fc(r_out)
        
        return output, hidden



