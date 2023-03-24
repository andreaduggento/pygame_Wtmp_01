import pygame

from main.entities.Entity import Entity
from main.utils.colors import *


class Boundary(Entity):
    def __init__(self, simulation, position, name):
        super().__init__(simulation, position, name)
        #self.load_image()
        self.color  = BLACK

class RoundBoundary(Boundary):
    def __init__(self, simulation, position, radius, name):
        super().__init__(simulation, position, name)
        #self.load_image()
        self.radius = radius

