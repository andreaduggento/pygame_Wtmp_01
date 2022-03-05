from __future__ import annotations
import pygame
import numpy as np
from main.utils.utils import *  
from main.utils.colors import *

class Position:
    theta = None
    def __init__(self, x , y, theta):
        self.x = x
        self.y = y
        self.theta = theta

class Entity:
    def __init__(self, simulation, position, name):
        self.name = name
        self.simulation = simulation
        self.screen = simulation.world.screen
        self.position = 1.*np.asarray(position)
        self.images_loaded = False
        self.image = None
        self.color = WHITE
        self.radius =0.
    
    def get_sizes(self):
        if self.images_loaded:
           return self.image.get_size() 
        else:
            return 2*[ self.radius ,self.radius ]

    def relative_angle_to(self,entity):
        diff = entity.position - self.position
        return np.arctan2(diff[1] , diff[0])   # always between -pi and pi

    def draw(self, world):
        pygame.draw.circle(world.screen, self.color , [self.position[0], world.size[1]-self.position[1]] , self.radius)
        if self.images_loaded:
            surf.blit(self.image, [self.position[0], world.size[1]-self.position[1]] )
        return self


class OrientedEntity(Entity):
    def __init__(self, simulation, position, name):
        super().__init__(simulation, position, name)
        self.orientation = 0.  # Angle expressed in radiants

    def draw(self, world):
        pygame.draw.circle(world.screen, self.color , [self.position[0], world.size[1]-self.position[1]] , self.radius)
        blitRotateBottomLeftRef(world.screen, self.image, self.position, self.get_sizes() , np.degrees(self.orientation), world.size[0] , world.size[1] )


