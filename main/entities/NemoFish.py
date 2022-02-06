import random
import pygame
from main.entities.Agents import *


class NemoFish(InteractiveAgent):

    MAX_CAPACITY = 1

    def __init__(self, simulation, position, name):
        super().__init__(simulation, position, name)
        self.load_image()
        self.MAX_OMEGA= simulation.DeltaT * 500.
        self.MAX_FORCE=100.

    def load_image(self):
        self.image = pygame.image.load("main/images/prototype_A01_32.png")
        self.images_loaded = True
        self.radius = 0.5*self.get_sizes()[0] 

    def draw(self, world):
        #super().draw(world)
        pygame.draw.circle(world.screen, (30,30,30), [self.position[0], world.size[1]-self.position[1]] , self.radius)

 
