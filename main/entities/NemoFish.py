import random
import pygame
from main.entities.Agents import *
from main.utils.colors import *


class NemoFish(InteractiveAgent):

    MAX_CAPACITY = 1

    def __init__(self, simulation, position, name):
        super().__init__(simulation, position, name)
        self.load_image()
        self.MAX_OMEGA= simulation.DeltaT * 500.
        self.MAX_FORCE=100.
        self.color = (130,130,130)
        self.energy = 0.

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

 
