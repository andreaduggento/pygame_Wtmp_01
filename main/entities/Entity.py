from __future__ import annotations
import math 
import pygame
import numpy as np
from main.utils.utils import *  
from main.utils.colors import *

class Position:
    theta = None
    def __init__(self, Tx, Ty, x , y, theta):
        self.x = x
        self.y = y
        self.theta = theta
class TilePosition:
    def __init__(self, Tx, Ty):
        self.Tx = Tx
        self.Ty = Ty
class Entity:
    def __init__(self, simulation, tileposition, position, name):
        self.name = name
        self.simulation = simulation
        self.screen = simulation.world.screen
        self.tileposition = np.asarray(tileposition)
        self.position = 1.*np.asarray(position)
        self.screenposition = 1.*np.asarray([ self.position[0] + simulation.world.tilesize * self.tileposition[0] , simulation.world.get_world_height() - self.position[1] - simulation.world.tilesize*self.tileposition[1] ])
        self.images_loaded = False
        self.image = None
        self.color = WHITE
        self.radius =0.

    def get_sizes(self):
        if self.images_loaded:
           return self.image.get_size() 
        else:
            return 2*[ self.radius ,self.radius ]

    def set_tileposition(self,tileposition):
        self.tileposition = np.asarray(tileposition)

    def set_position(self,position):
        self.position = 1.*np.asarray(position)

    def relative_angle_to(self,entity):                     #DA FARE RISCRIVERE
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

    def relative_distance_to(self,entity):                     #DA FARE RISCRIVERE
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

    def relative_distanceNORM_to(self,entity):                     #DA FARE RISCRIVERE
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
        pygame.draw.circle(world.screen, self.color , self.screenposition , self.radius)
        if self.images_loaded:
            surf.blit(self.image, self.screenposition )
        return self


class OrientedEntity(Entity):
    def __init__(self, simulation, tileposition, position, name):
        super().__init__(simulation, tileposition, position, name)
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
        pygame.draw.circle(world.screen, self.color , self.screenposition , self.radius)
        blitRotateBottomLeftRef(world.screen, self.image, [0., world.get_world_height()] + [1.,-1.]*self.screenposition, self.get_sizes() , np.degrees(self.orientation), world.get_world_width() , world.get_world_height() )


