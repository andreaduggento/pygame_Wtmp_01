from __future__ import annotations
import pygame
import numpy as np
from main.utils.utils import *  

class Position:
    theta = None
    def __init__(self, x , y, theta):
        self.x = x
        self.y = y
        self.theta = theta



class Entity:
    radius = 0
    def __init__(self, simulation, position, name):
        self.name = name
        self.simulation = simulation
        self.screen = simulation.world.screen
        self.position = 1.*np.asarray(position)
        self.images_loaded = False
        self.image = None
    
    def get_sizes(self):
        if self.images_loaded:
           return self.image.get_size() 
        else:
            return 2*[ self.radius ,self.radius ]

    def relative_angle_to(self,entity):
        diff = entity.position - self.position
        return np.arctan2(diff[1] , diff[0])   # always between -pi and pi

class OrientedEntity(Entity):
    def __init__(self, simulation, position, name):
        super().__init__(simulation, position, name)
        self.orientation = 0.  # Angle expressed in radiants

    def draw(self, world):
        blitRotateBottomLeftRef(world.screen, self.image, self.position, self.get_sizes() , np.degrees(self.orientation), world.size[0] , world.size[1] )



