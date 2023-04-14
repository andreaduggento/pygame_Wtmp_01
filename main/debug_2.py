import sys
import pygame
import math 
import time
import random
import string
import math
import numpy as np
import pygame
from main.utils.colors import *
from main.utils.utils import *

from main.entities.Entity import Entity,OrientedEntity
from main.brains.Brains import *


sys.path.append('./')
from main.world.Simulation import *


if __name__ == "__main__":
    
    INPUT_SIZE = 3
    OUTPUT_SIZE = 4
    SEQ_LENGTH = 1
    HIDDEN_SIZE = 6
    OUTPUT_SIZE = 3
    NUM_LAYERS = 1
    BATCH_SIZE = 1

    brain = spRNN(3,3)
            sprnnBrain(INPUT_SIZE,OUTPUT_SIZE,parentnet=None):


#    x = np.zeros(self.neyes, dtype=float)
#    x = np.random.normal(loc=0.0, scale=.01, size=(INPUT_size)) 


 #   x = self.brain.forward(torch.from_numpy(self.perception))
    


