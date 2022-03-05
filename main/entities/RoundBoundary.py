import pygame

from main.entities.Entity import Entity
from main.utils.colors import *


class RoundBoundary(Entity):

    def __init__(self, simulation, position, radius, name):
        super().__init__(simulation, position, name)
        #self.load_image()
        self.radius = radius
        self.color  = BLACK

