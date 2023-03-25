import torch
import torch.nn as nn

class annBrain():

    def __init__(self,n_input,n_output):
        self.n_input=n_input
        self.n_output=n_output

        self.model = nn.Sequential(nn.Linear(n_input,n_output),
                      nn.Sigmoid())
        print(self.model)

