import random
import pygame
from main.entities.Agents import *
from main.utils.colors import *


class NemoFish(InteractiveAgent):

    MAX_CAPACITY = 1

    def __init__(self, simulation, tileposition, position, name):
        super().__init__(simulation, tileposition, position, name)

    def load_image(self):
        self.image = pygame.image.load("main/images/prototype_A01_32.png")
        self.images_loaded = True
        self.radius = 0.5*self.get_sizes()[0] 

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
        self.eatpollen(sim)
        super().update(sim)


class NemoBrain(rnnAgent):

    MAX_CAPACITY = 1

    def __init__(self, simulation, tileposition, position, name):
        super().__init__(simulation, tileposition, position, name)
        self.vel = np.asarray([0.,0.])
        self.acc = np.asarray([0.,0.])
        self.force= 0.
        self.mass = 1.
        self.omega = 0.
        self.color = WHITE
        self.load_image()
        self.MAX_OMEGA= simulation.DeltaT * 50.
        self.MAX_FORCE=50.
        self.color = (130,130,130)
        self.energycolor = (130,130,130)
        self.DefaultEnergy = 0.5               
        self.energy = self.DefaultEnergy               # from 0 to 1
        self.metabolic_rate = 0.00001                               # basal energy consumption per unit of DeltaT
        self.metabolicspeed = self.metabolic_rate / self.MAX_FORCE  # active energy consumption per unit of force, per unit of DeltaT
        self.metabolicomega = self.metabolic_rate                # active energy consumption per unit of force, per unit of omega
        self.energyTotInOut = np.zeros(4)
        self.fertility = False
        ## controls
        self.Kleft=False
        self.Kright=False
        self.Kup=False
        self.Kdown = False
        

    def update(self,sim): 
        self.interract_with_pollens(sim)
        self.actuate()
        self.energyTotInOut = self.energyTotInOut +  sim.DeltaT*np.asarray([self.metabolic_rate, np.linalg.norm(self.force)*self.metabolicspeed,abs(self.omega)*self.metabolicomega,0.])
        self.energy -= ( self.metabolic_rate +  np.linalg.norm(self.force)*self.metabolicspeed + abs(self.omega)*self.metabolicomega  )  * sim.DeltaT 
        self.updatecolor()
        super().update(sim)

    def actuate(self):
        x = self.brain.forward(torch.from_numpy(self.perception))
        print(x)

    def reproduce(self,sim):
        return

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


  
