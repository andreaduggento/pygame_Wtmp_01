from __future__ import annotations
import math 
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

#    def __del__(self):
#        print("Deleting entity")
#        super().__del__()


    def get_sizes(self):
        if self.images_loaded:
           return self.image.get_size() 
        else:
            return 2*[ self.radius ,self.radius ]

##    def relative_angle_to(self,entity):
##        diff = entity.position - self.position
##        return np.arctan2(diff[1] , diff[0])   # always between -pi and pi

    def set_position(self,position):
        self.position = 1.*np.asarray(position)

    def relative_angle_to(self,entity):
        diff = entity.position - self.position
        if (diff[0]>0.5*self.simulation.world.size[0]):
            diff[0]=diff[0]-self.simulation.world.size[0]
        if (diff[0]<-0.5*self.simulation.world.size[0]):
            diff[0]=diff[0]+self.simulation.world.size[0]
        if (diff[1]>0.5*self.simulation.world.size[1]):
            diff[1]=diff[1]-self.simulation.world.size[1]
        if (diff[1]<-0.5*self.simulation.world.size[1]):
            diff[1]=diff[1]+self.simulation.world.size[1]
        return np.arctan2(diff[1] , diff[0])   # always between -pi and pi

    def relative_distance_to(self,entity):
        diff = entity.position - self.position
        if (diff[0]>0.5*self.simulation.world.size[0]):
            diff[0]=diff[0]-self.simulation.world.size[0]
        if (diff[0]<-0.5*self.simulation.world.size[0]):
            diff[0]=diff[0]+self.simulation.world.size[0]
        if (diff[1]>0.5*self.simulation.world.size[1]):
            diff[1]=diff[1]-self.simulation.world.size[1]
        if (diff[1]<-0.5*self.simulation.world.size[1]):
            diff[1]=diff[1]+self.simulation.world.size[1]
        return diff 

    def relative_distanceNORM_to(self,entity):
        diff = entity.position - self.position
        if (diff[0]>0.5*self.simulation.world.size[0]):
            diff[0]=diff[0]-self.simulation.world.size[0]
        if (diff[0]<-0.5*self.simulation.world.size[0]):
            diff[0]=diff[0]+self.simulation.world.size[0]
        if (diff[1]>0.5*self.simulation.world.size[1]):
            diff[1]=diff[1]-self.simulation.world.size[1]
        if (diff[1]<-0.5*self.simulation.world.size[1]):
            diff[1]=diff[1]+self.simulation.world.size[1]
        return diff/ np.linalg.norm(diff)

    def draw(self, world):
        pygame.draw.circle(world.screen, self.color , [self.position[0], world.size[1]-self.position[1]] , self.radius)
        if self.images_loaded:
            surf.blit(self.image, [self.position[0], world.size[1]-self.position[1]] )
        return self


class OrientedEntity(Entity):
    def __init__(self, simulation, position, name):
        super().__init__(simulation, position, name)
        self.orientation = 0.  # Angle expressed in radiants

    def relative_normcross_to(self,entity):
        differenceNORM  = self.relative_distanceNORM_to(entity)
        selforientNORM  = np.array([ math.cos(self.orientation) ,  math.sin( self.orientation ) ])
        return np.cross(selforientNORM,differenceNORM)    # always between -1 and 1 

    def relative_normdot_to(self,entity):
        differenceNORM  = self.relative_distanceNORM_to(entity)
        selforientNORM  = np.array([ math.cos(self.orientation) ,  math.sin( self.orientation ) ])
        return np.dot(selforientNORM,differenceNORM)    # always between -1 and 1 

    def relative_biased_normdot_to(self,entity,eyesradpos,threshold):
        differenceNORM  = self.relative_distanceNORM_to(entity)
        output = np.zeros(len(eyesradpos))
        for i,bias in enumerate(eyesradpos):
            biasedselforientNORM  = np.array([ math.cos(self.orientation+bias) ,  math.sin( self.orientation+bias ) ])
            output[i] = np.dot(biasedselforientNORM , differenceNORM)  
        output[output < threshold] = 0
        return output    # always between -1 and 1 

    def draw(self, world):
        pygame.draw.circle(world.screen, self.color , [self.position[0], world.size[1]-self.position[1]] , self.radius)
        blitRotateBottomLeftRef(world.screen, self.image, self.position, self.get_sizes() , np.degrees(self.orientation), world.size[0] , world.size[1] )


