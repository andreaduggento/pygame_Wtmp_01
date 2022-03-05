import random
import pygame
import numpy as np
from main.utils.utils import *  
from main.entities.Agents import *
from main.utils.colors import *


class Follower1(Agent):
    def __init__(self, simulation, position, name):
        super().__init__(simulation, position, name)
        self.load_image()
        self.MAX_OMEGA=100.
        self.MAX_FORCE=10.
        self.orientation=0.
        self.speed = 30.
        self.color = PALERED
    def update(self,sim):
#        super().update(sim)
#        self.update_angle(sim)
        self.orientation = self.relative_angle_to(sim.target)
        self.vel         = self.speed * np.array([ math.cos (self.orientation) , math.sin(self.orientation) ]) 
        self.position    += self.vel * sim.DeltaT
        ## Keep distance from other sharks
        # self.keep_distance(sim)  
        ## Always check boundaries
        self.checkboundaries(sim)

    def load_image(self):
        self.image = pygame.image.load("main/images/prototype_A02_32.png")
        self.images_loaded = True
        self.radius = 0.5*self.get_sizes()[0] 

class Flocker1(Agent):
    def __init__(self, simulation, position, name):
        super().__init__(simulation, position, name)
        self.load_image()
        self.MAX_OMEGA=100.
        self.MAX_FORCE=10.
        self.orientation=0.
        self.speed = 30.
        self.color = PALERED
        self.MAX_OMEGA= simulation.DeltaT * 500.
        self.MAX_FORCE=10.
        self.vel = np.asarray([0.,0.])
        
        self.force = self.MAX_FORCE * np.array([1.,0])
     
    def update(self,sim):
        self.keep_distance(sim)  
        super().update(sim)

    def keep_distance(self,sim):
        for other in sim.agents :
            if isinstance(other,Flocker1):
                pass
#                distance = np.linalg.norm(other.position - self.position) - (other.radius + self.radius)
#                if distance < 3*self.radius :
#                # updade position
#                    angle = self.relative_angle_to(other)
#                    diff = (other.position - self.position) - (other.radius + self.radius)
#                    # print(angle)
#                    self.vel      = self.vel - 2/(distance) * np.array( [ math.cos(angle)*abs(self.vel[0]) , math.sin( angle )*abs(self.vel[1])  ])
#

    def load_image(self):
        self.image = pygame.image.load("main/images/prototype_A02_32.png")
        self.images_loaded = True
        self.radius = 0.5*self.get_sizes()[0] 


