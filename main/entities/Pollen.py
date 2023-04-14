import pygame

from main.entities.Entity import Entity
from main.utils.colors import *
# from main.agents.ReactiveAgent import ReactiveAgent
# from main.utils.Pathfinding import path_finding


class Pollen(Entity):

    def __init__(self, simulation, tileposition , position, name):
        super().__init__(simulation, tileposition, position, name)
        #self.load_image()
        self.radius = 5.
        self.color  = PALERED
        self.energy = 0.05

    def update(self,sim):
        self.eatpollen(sim)
        super().update(sim)

    def make_visible(self):
        self.color  = RED
        
    def make_invisible(self):
        self.color  = PALERED

    def get_energy(self):
        return self.energy

#    def draw(self, world):
#        pygame.draw.circle(world.screen, RED , [self.position[0], world.size[1]-self.position[1]] , self.radius)
#        return self

