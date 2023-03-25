import math 
import numpy as np
import pygame
from main.utils.colors import *

from main.entities.Entity import Entity,OrientedEntity
from main.brains.Brains import *
# from main.agents.ReactiveAgent import ReactiveAgent
# from main.utils.Pathfinding import path_finding


class Agent(OrientedEntity):
    def __init__(self, simulation, position, name):
        super().__init__(simulation, position, name)
        self.vel = np.asarray([0.,0.])
        self.acc = np.asarray([0.,0.])
        self.force= 0.
        self.MAX_OMEGA = 10.
        self.MAX_FORCE = 10.
        self.mass = 1.
        self.omega = 0.
        self.color = WHITE
        self.load_image()
        self.MAX_OMEGA= simulation.DeltaT * 500.
        self.MAX_FORCE=100.
        self.color = (130,130,130)
        self.energy = 0.


    def update(self,sim):
        while self.orientation > math.pi:
            self.orientation -= 2*math.pi
        while self.orientation < -2*math.pi:
            self.orientation += 2*math.pi
        self.acc         = self.force/self.mass  - sim.FRICTION * self.vel
        self.vel         += sim.DeltaT * self.acc
        self.orientation += self.omega*sim.DeltaT
        self.position    += self.vel * sim.DeltaT
        if sim.worldtorus:
            for i in range(0,2):
                if self.position[i] < 0:
                    self.position[i] += sim.size[i]
                elif self.position[i] > sim.size[i]:
                    self.position[i] -= sim.size[i]
        ## Always check boundaries
#        self.checkboundaries(sim)
        self.bounch(sim)
        return self

    def load_image(self):
        self.image = pygame.image.load("main/images/prototype_A01_32.png")
        self.images_loaded = True
        self.radius = 0.5*self.get_sizes()[0] 


    def bounch(self,sim):
        for BOU in sim.agents :
            if id(BOU)!=id(self):
                distance = np.linalg.norm(BOU.position - self.position) - (BOU.radius + self.radius)
                if distance < 0 :
                    angle = self.relative_angle_to(BOU)
                    diff = (BOU.position - self.position) - (BOU.radius + self.radius)
                    self.position = self.position + distance*np.array( [ math.cos(angle) ,  math.sin(angle) ])
                    self.vel      = self.vel - 2 * np.array( [ math.cos(angle)*abs(self.vel[0]) , math.sin( angle )*abs(self.vel[1])  ])
        for BOU in sim.roundboundaries :
            if id(BOU)!=id(self):
                distance = np.linalg.norm(BOU.position - self.position) - (BOU.radius + self.radius)
                if distance < 0 :
                    angle = self.relative_angle_to(BOU)
                    diff = (BOU.position - self.position) - (BOU.radius + self.radius)
                    self.position = self.position + distance*np.array( [ math.cos(angle) ,  math.sin(angle) ])
                    self.vel      = self.vel - 2 * np.array( [ math.cos(angle)*abs(self.vel[0]) , math.sin( angle )*abs(self.vel[1])  ])



##     def checkboundaries(self,sim):
##         for BOU in sim.roundboundaries :
##             distance = np.linalg.norm(BOU.position - self.position) - (BOU.radius + self.radius)
##             if distance < 0 :
##                 print("bounching")
##                 # updade position
##                 angle = self.relative_angle_to(BOU)
##                 diff = (BOU.position - self.position) - (BOU.radius + self.radius)
##                 print(angle)
## #                self.position = self.position - np.array( [ math.cos(angle)*abs((BOU.position[0] - self.position[0]))   ,  math.sin(angle)*abs(BOU.position[1] - self.position[1]) ])
##                 self.position = self.position + distance*np.array( [ math.cos(angle) ,  math.sin(angle) ])
##                 self.vel      = self.vel - 2 * np.array( [ math.cos(angle)*abs(self.vel[0]) , math.sin( angle )*abs(self.vel[1])  ])


class InteractiveAgent(Agent):
    def __init__(self, simulation, position, name):
        super().__init__(simulation, position, name)
        self.Kleft=False
        self.Kright=False
        self.Kup=False
        self.Kdown = False

    def process_event(self,event):
        ############################
        if event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
            self.Kleft=True
            self.Kright=False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
            self.Kleft=False
            self.Kright=True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
            self.Kup=False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
            self.Kdown = True
        if event.type == pygame.KEYUP and event.key == pygame.K_UP:
            self.Kup = False
        if event.type == pygame.KEYUP and event.key == pygame.K_DOWN:
            self.Kdown = False
        if event.type == pygame.KEYUP and event.key == pygame.K_LEFT : 
            self.Kleft=False
        if event.type == pygame.KEYUP and event.key == pygame.K_RIGHT:
            self.Kright=False
        #############    IFs
        if self.Kleft :
            self.omega = self.MAX_OMEGA
        elif  self.Kright :
            self.omega = -self.MAX_OMEGA
        else :
            self.omega = 0.  
        if self.Kdown :
            fire=True
            self.force =  np.array([ math.cos(self.orientation)*self.MAX_FORCE ,  math.sin( self.orientation )*self.MAX_FORCE ])
        else:
            fire=False
            self.force =  np.array([0.,0.]) 
            ######################


class IntelligentAgent(Agent):

    def __init__(self, simulation, position, name):
        super().__init__(simulation, position, name)
        self.brain = annBrain(3,3)
        self.omega=0.05*self.MAX_OMEGA
        self.neyes = 3
        self.halfaperture = 0.7     # in radiants
        self.eyesthreshold = 0.7    # threshold from 0 to 1
        self.eyesradpos = np.zeros(self.neyes, dtype=float)
#        self.eyespos = np.zeros((self.neyes,2), dtype=float)
        self.init_eyes_pos(self.neyes,self.halfaperture) 


    def init_eyes_pos(self,neyes,halfaperture): 
        # halfaperture is the angle displacement of outermost eye expressed in rad
        for i in range(self.neyes):
            self.eyesradpos[i] = -halfaperture + 2.*(i/(self.neyes-1))*halfaperture
#            self.eyespos[i] = np.array([math.cos(-halfaperture + 2.*(i/(self.neyes-1))*halfaperture),math.sin(-halfaperture + 2.*(i/(self.neyes-1))*halfaperture) ])
        print(self.eyesradpos)

    def load_image(self):
        self.image = pygame.image.load("main/images/prototype_A04_32.png")
        self.images_loaded = True
        self.radius = 0.5*self.get_sizes()[0] 

    def __del__(self):
        print("Deleting intelligent agent"+self.name)

    def eatpollen(self,sim):
        for pollen in sim.pollens :
            distance = np.linalg.norm(pollen.position - self.position) - (pollen.radius + self.radius)
            if distance < 0 :
                self.energy = self.energy + 1.
                self.updatecolor()
                sim.remove_entity(pollen)

    def decrease_energy(self,energy):
            self.energy = self.energy - energy

    def updatecolor(self):
        self.color = (130,max(0,min(255,130+10*self.energy)),130)

    def update(self,sim):
        self.perceivepollen(sim)
        super().update(sim)

    def perceivepollen(self,sim):
        for pollen in sim.pollens :
            distance = np.linalg.norm(pollen.position - self.position) - (pollen.radius + self.radius)
            if distance < 50 :
                print(self.relative_biased_normdot_to(pollen,self.eyesradpos,self.eyesthreshold))
            



