import random
import pygame
import numpy as np
from main.utils.utils import *  
from main.entities.Agents import *


class Follower1(Agent):

    def __init__(self, simulation, position, name):
        super().__init__(simulation, position, name)
        self.load_image()
        self.MAX_OMEGA=100.
        self.MAX_FORCE=10.
        self.orientation=0.
        self.speed = 30.


    def update(self,sim):
#        super().update(sim)
#        self.update_angle(sim)
        self.orientation = self.relative_angle_to(sim.target)
        self.vel         = self.speed * np.array([ math.cos (self.orientation) , math.sin(self.orientation) ]) 
        self.position    += self.vel * sim.DeltaT


#    def update_angle(self,sim):
#        anglediff = ( self.relative_angle_to(sim.target) - self.orientation )
#        diff = anglediff
#        if abs(anglediff - math.pi) < anglediff:
#            diff = anglediff - 2*math.pi
#        elif abs(anglediff + math.pi) < anglediff:
#            diff = anglediff + 2*math.pi
#        self.omega = self.MAX_OMEGA * sim.DeltaT * diff
#        if abs(anglediff< .6 ):
#            self.force = self.MAX_FORCE * np.array([ math.cos (self.orientation) , math.sin(self.orientation) ])


    def load_image(self):
        self.image = pygame.image.load("main/images/prototype_A02_32.png")
        self.images_loaded = True


