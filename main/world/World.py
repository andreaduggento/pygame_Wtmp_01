import random
from math import sqrt
import pygame
from main.utils.colors import *

def distance_between(a, b):
    return sqrt(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))




font_name = pygame.font.match_font('arial')

## def toroidal_distance(a,b):
##     dx = abs(a[0] - b[0]);
##     dy = abs(a[1] - b[1]);
##  
##     return std::sqrt(dx*dx + dy*dy);
## }


class World:

    def __init__(self,worldsize):
        self.size = worldsize
        self.screen = pygame.display.set_mode((self.get_world_width(), self.get_world_height()))


    def get_world_width(self):
        return self.size[0]

    def get_world_height(self):
        return self.size[1]

    def draw(self, simulation, entities):
        self.screen.fill(WHITE)
        for entity in entities:
            entity.draw(self)
