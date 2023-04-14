import math 
import numpy as np
import pygame
from main.utils.colors import *
from main.utils.utils import *

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
        self.mass = 1.
        self.omega = 0.
        self.color = WHITE
        self.load_image()
        self.MAX_OMEGA= simulation.DeltaT * 500.
        self.MAX_FORCE=100.
        self.color = (130,130,130)
        self.energycolor = (130,130,130)
        self.DefaultEnergy = 0.5               
        self.energy = self.DefaultEnergy               # from 0 to 1
        self.metabolic_rate = 0.01                               # basal energy consumption per unit of DeltaT
        self.metabolicspeed = self.metabolic_rate / self.MAX_FORCE  # active energy consumption per unit of force, per unit of DeltaT
        self.metabolicomega = self.metabolic_rate                # active energy consumption per unit of force, per unit of omega
        self.energyTotInOut = np.zeros(4)
        self.fertility = True

    def update(self,sim):
        if (self.energy < 0.):
            self.die(sim)
        if self.fertility:
            if (self.energy > 2*self.DefaultEnergy):
                self.energy -= self.DefaultEnergy
                sim.agent_born(self)

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

    def get_energy(self):
        return self.energy
        
    def get_energyTotInOut(self):
        return self.energyTotInOut
 
    def set_default_energy(self):
        self.energy = self.DefaultEnergy
 
    def set_energy(self,energy):
        self.energy = energy 

    def load_image(self):
        self.image = pygame.image.load("main/images/prototype_A01_32.png")
        self.images_loaded = True
        self.radius = 0.5*self.get_sizes()[0] 

    def die(self,sim):
        sim.agent_die(self)

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

    def reproduce(self,sim):
        offspring = Agent(sim,self.position, "agent_"+randomname(10))
        return offspring



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

    def __init__(self, simulation, position, name, brain=None ):
        super().__init__(simulation, position, name)
        self.omega=0.05*self.MAX_OMEGA
        self.neyes = 3
        self.eyerange = 150 
        self.perception = np.zeros(self.neyes, dtype=float)
        self.halfaperture = 0.7     # in radiants
        self.eyesthreshold = 0.75    # threshold from 0 to 1
        self.eyesradpos = np.zeros(self.neyes, dtype=float)
#        self.eyespos = np.zeros((self.neyes,2), dtype=float)
        self.init_eyes_pos(self.neyes,self.halfaperture) 
        self.showvisibility = False


    def update(self,sim):
        self.interract_with_pollens(sim)
        self.actuate()
        self.energyTotInOut = self.energyTotInOut +  sim.DeltaT*np.asarray([self.metabolic_rate, np.linalg.norm(self.force)*self.metabolicspeed,abs(self.omega)*self.metabolicomega,0.])
        self.energy -= ( self.metabolic_rate +  np.linalg.norm(self.force)*self.metabolicspeed + abs(self.omega)*self.metabolicomega  )  * sim.DeltaT 
        self.updatecolor()
        super().update(sim)

    def init_eyes_pos(self,neyes,halfaperture): 
        # halfaperture is the angle displacement of outermost eye expressed in rad
        for i in range(self.neyes):
            self.eyesradpos[i] = -halfaperture + 2.*(i/(self.neyes-1))*halfaperture
#            self.eyespos[i] = np.array([math.cos(-halfaperture + 2.*(i/(self.neyes-1))*halfaperture),math.sin(-halfaperture + 2.*(i/(self.neyes-1))*halfaperture) ])
#        print(self.eyesradpos)

    def load_image(self):
        self.image = pygame.image.load("main/images/prototype_A04_32.png")
        self.images_loaded = True
        self.radius = 0.5*self.get_sizes()[0] 

#    def __del__(self):
#        print("Deleting intelligent agent"+self.name)

    def eatpollen(self,sim,pollen):
            self.energy = self.energy + pollen.get_energy() 
            self.energyTotInOut +=  np.asarray([ 0. , 0. , 0.  , pollen.get_energy() ]) 
            sim.pollen_eaten(pollen)


    def decrease_energy(self,energy):
            self.energy = self.energy - energy

    def updatecolor(self):
#        self.energycolor = ( 130 , max(0 , min( 255 ,  255 * self.energy )) , 130 )
        self.energycolor = ( max(50 , min( 200 ,  200 * (1-self.energy) ))  , max(50 , min( 200 ,  200 * self.energy )) , 30 )


    def draw(self, world):
        self.color = self.energycolor
        super().draw(world)
        #pygame.draw.circle(world.screen, BLACK , [self.position[0], world.size[1]-self.position[1]] , 6)
        #pygame.draw.circle(world.screen, self.energycolor , [self.position[0], world.size[1]-self.position[1]] , 5)


    def perceivepollen(self,sim,pollen,distance):
            if (self.showvisibility): pollen.make_invisible()
            if distance < self.eyerange :
                tmp =  (  2. * pollen.radius * self.relative_biased_normdot_to(pollen,self.eyesradpos,self.eyesthreshold) / (distance+0.0001) )
                if ( tmp.sum() > 0.000000000001 ):
                    if (self.showvisibility): pollen.make_visible()
                self.perception = self.perception + tmp
                # self.perception = self.perception + (10* self.relative_biased_normdot_to(pollen,self.eyesradpos,self.eyesthreshold) / distance )

    def interract_with_pollens(self,sim):
        self.perception[:] = 0.
        for pollen in sim.pollens :
            distance = np.linalg.norm(pollen.position - self.position) - (pollen.radius + self.radius)
            if distance < 0 :
                self.eatpollen(sim,pollen) 
            else:
                self.perceivepollen(sim,pollen,distance)

    def actuate(self):
        x = self.brain.forward(torch.from_numpy(self.perception))
        force = x[0].item()
        left  = x[1].item()
        right=  x[2].item()
        # print(left,force,right)
        self.omega = self.MAX_OMEGA * (right - left)     
        if (force > 0.):
            self.force =  force * np.array([ math.cos(self.orientation)*self.MAX_FORCE ,  math.sin( self.orientation )*self.MAX_FORCE ])

    def print_agent(self):
        print("agent "+ self.name+"; energy={}".format(round(self.energy,4)) + " force={}".format( self.force)  + " omega={}".format(round(self.omega,4)))

    def print_to_file(self,file1):
        L = ["Agent: "+ self.name + " Energy components: [" +  ' '.join(map(str, self.get_energyTotInOut())) + "] Current energy: " +  ' {}'.format(self.get_energy()) + '\n' ]
        file1.writelines(L)
        self.print_brain_to_file(file1)

    def print_brain(self):
        self.brain.printbrain()

    def print_brain_to_file(self,file1):
        self.brain.printbrain_to_file(file1)


    def turnvisibility(self,visibility):
        self.showvisibility = visibility


class annAgent(IntelligentAgent):
    def __init__(self, simulation, position, name, brain=None ):
        if brain==None : 
            self.brain = annBrain(3,3)
        else: 
            self.brain = brain
        super().__init__(simulation, position, name)

    def load_image(self):
        self.image = pygame.image.load("main/images/prototype_A04_32.png")
        self.images_loaded = True
        self.radius = 0.5*self.get_sizes()[0] 

    def reproduce(self,sim):
        newbrain = self.brain.reproduce()
        offspring = annAgent(sim,self.position, "annNemo_"+randomname(10),newbrain)
        return offspring

class rnnAgent(IntelligentAgent):
    def __init__(self, simulation, position, name, brain=None ):
        if brain==None : 
            self.brain = rnnBrain(3,3)
        else: 
            self.brain = brain
        super().__init__(simulation, position, name)

    def load_image(self):
        self.image = pygame.image.load("main/images/prototype_A02_32.png")
        self.images_loaded = True
        self.radius = 0.5*self.get_sizes()[0] 


    def reproduce(self,sim):
        newbrain = self.brain.reproduce()
        offspring = rnnAgent(sim,self.position,  "rnnNemo_"+randomname(10),newbrain)
        return offspring
 

    def draw(self, world):
        super().draw(world)
        pygame.draw.circle(world.screen, BLACK , [self.position[0], world.size[1]-self.position[1]] , 6)
        pygame.draw.circle(world.screen, self.energycolor , [self.position[0], world.size[1]-self.position[1]] , 4)


