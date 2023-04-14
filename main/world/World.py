import random
from math import sqrt
import pygame
from main.utils.colors import *

def distance_between(a, b):
    return sqrt(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


font_name = pygame.font.match_font('arial')



class World:

    def __init__(self,tilesize,worldsize):
        self.tilesize= tilesize
        self.size = worldsize       # in tiles
        self.screen = pygame.display.set_mode((self.get_world_width(), self.get_world_height()))
        self.grid = True

    def get_world_width(self):
        return self.size[0]*self.tilesize

    def get_world_height(self):
        return self.size[1]*self.tilesize

    def draw(self, simulation, entities):
        self.screen.fill(WHITE)
        if self.grid == True:
            for i in range (self.size[0]):
                pygame.draw.line(self.screen, BLACK , (i*self.tilesize, 0 ) , (i*self.tilesize,self.get_world_height()))
            for j in range (self.size[1]):
                pygame.draw.line(self.screen, BLACK , (0 , j*self.tilesize ) , (self.get_world_width(),j*self.tilesize,))
        for entity in entities:
            entity.draw(self)
