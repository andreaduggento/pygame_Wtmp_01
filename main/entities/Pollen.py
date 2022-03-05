import pygame

from main.entities.Entity import Entity
from main.utils.colors import *
# from main.agents.ReactiveAgent import ReactiveAgent
# from main.utils.Pathfinding import path_finding


class Pollen(Entity):

    def __init__(self, simulation, position, name):
        super().__init__(simulation, position, name)
        #self.load_image()
        self.radius = 5
        self.color  = RED

#    def draw(self, world):
#        pygame.draw.circle(world.screen, RED , [self.position[0], world.size[1]-self.position[1]] , self.radius)
#        return self

