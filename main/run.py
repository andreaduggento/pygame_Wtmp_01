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
    pygame.display.set_caption("Ediacaran Zoo")

    # Form screen
    screen = pygame.display.set_mode()
 
    # get the default size
    x, y = screen.get_size()
    WORLDSIZE=[x-10,y-50]
    print(WORLDSIZE)
#    simulation = SimulationFlock(WORLDSIZE)
    simulation = EdiacaranZoo(WORLDSIZE)
    
    counter = 0
    
    # Run
#    while counter < 10000:
    GO = True
    while GO:
        simulation.update()
        pygame.display.update()
        if ((counter % 50) == 0) :
            simulation.print_best_agent()
            print("  N.Agents: {}".format(simulation.n_agent())+"   average energy: {}".format(round(simulation.average_agent_energy(),4)),end='\r')
        if (simulation.n_agent()<2):
            simulation.print_agents()
            GO=False
            gameExit = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                gameExit = True
            else:
                simulation.process_event(event)
        counter += 1

if __name__ == "__main__":
    start()
