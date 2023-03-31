import sys
import pygame
import math 
import time
import random
import string


sys.path.append('./')
from main.world.Simulation import *

def start():
    # Create world
    pygame.init()
    pygame.display.set_caption("Ediacaran Zoo")

    file1 = open("allagents_"+ time.strftime("%Y%m%d-%H%M%S")+ ".txt", "w")

    # Form screen
    screen = pygame.display.set_mode()
 
    # get the default size
    x, y = screen.get_size()
    WORLDSIZE=[x-100,y-100]
    print(WORLDSIZE)
#    simulation = SimulationFlock(WORLDSIZE)
    simulation = EdiacaranZoo(WORLDSIZE, agentstatFile = file1 ,  printreproductiveAgent = True )
    
    counter = 0
    
    # Run
#    while counter < 10000:
    GO = True
    while GO:
        simulation.update()
        pygame.display.update()
        if ((counter % 50) == 0) :
            simulation.print_best_agent()
            print(" count:{}".format(counter)+"  N.Agents: {}".format(simulation.n_agent())+"   total energy: {}".format(round(simulation.total_agent_energy(),5)) +" [threshold:{:.4f}".format(simulation.embodied_energy),end=' \r')
        #if ((counter % 5000) == 0) :
        #    print("---------------------------------------------------------------DECIMATION------------------------------------------------")
        #    simulation.decimation()
        #    print(" count:{}".format(counter)+"  N.Agents: {}".format(simulation.n_agent())+"   average energy: {}".format(round(simulation.average_agent_energy(),4)),end='\r')
        if (simulation.n_agent()<2):
            simulation.print_agents()
            GO=False
            gameExit = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                gameExit = True
            else:
                simulation.process_event(event)
            print(" count:{}".format(counter)+"  N.Agents: {}".format(simulation.n_agent())+"   total energy: {}".format(round(simulation.total_agent_energy(),5)) +" [threshold:{:.4f}".format(simulation.pollen_threshold),end=' \r')
        counter += 1

if __name__ == "__main__":
    start()
