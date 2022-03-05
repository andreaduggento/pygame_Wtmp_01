import random
import pygame
from main.entities.Agents import *
from main.utils.colors import *


class NemoFish(InteractiveAgent):

    MAX_CAPACITY = 1

    def __init__(self, simulation, position, name):
        super().__init__(simulation, position, name)
        self.load_image()
        self.MAX_OMEGA= simulation.DeltaT * 500.
        self.MAX_FORCE=100.
        self.color = PALEBLUE

    def load_image(self):
        self.image = pygame.image.load("main/images/prototype_A01_32.png")
        self.images_loaded = True
        self.radius = 0.5*self.get_sizes()[0] 


 
