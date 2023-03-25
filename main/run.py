import sys
import pygame
import math 
import time
import random



sys.path.append('./')
from main.world.Simulation import *



def start():
    # Create world
    pygame.init()
    pygame.display.set_caption("Worm sandbox")

    # Form screen
    screen = pygame.display.set_mode()
 
    # get the default size
    x, y = screen.get_size()
    WORLDSIZE=[x-300,y-100]
    print(WORLDSIZE)
#    simulation = SimulationFlock(WORLDSIZE)
    simulation = EdiacaranZoo(WORLDSIZE)
    
    counter =10000
    
    # Run
    while counter >0:
        simulation.update()
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                gameExit = True
            else:
                simulation.process_event(event)
        counter-=1

if __name__ == "__main__":
    start()
