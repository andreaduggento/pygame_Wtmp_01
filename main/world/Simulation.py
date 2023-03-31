from __future__ import annotations
import random
import string
import pygame
import time
from datetime import datetime, timedelta
from main.entities.Agents import *
from main.entities.Pollen import Pollen
from main.entities.RoundBoundary import RoundBoundary
from main.entities.NemoFish import *
from main.entities.Sharks import *
from main.world.World import World, distance_between
import numpy as np

class Simulation:

    SPOT_TIME_STEP = timedelta(seconds=1000.0)
    DeltaT = 0.02
    TIME_STEP = timedelta(seconds=DeltaT)
    VERBOSE = False
    FRICTION = 1.


    def __init__(self,worldsize, agentstatFile = None , decimationstatFile = None ,  printreproductiveAgent = True ):
        self.size=worldsize
        self.entities = []
        self.pollens = []
        self.roundboundaries = []
        self.agents = []
        self.interactiveagents = []
        self.world = World(worldsize)
        self.populate_zoo()
        self.next_spot_time = datetime.now()
        self.next_step_time = datetime.now()
        self.worldtorus = True
        self.current_step = 1

        ''' # statistics '''
        self.agentstatFile=agentstatFile
        self.decimation_number = 0
        self.printreproductiveAgent=printreproductiveAgent
        if printreproductiveAgent:
            self.bestagentfilename = "reproductiveagent_"+ time.strftime("%Y%m%d-%H%M%S")+ ".txt"

    def populate_zoo(self):
        return

    def add_entity(self, entity):
        if entity in self.entities:
            pass
        else:
            self.entities.append(entity)
        if isinstance(entity,RoundBoundary):
            if entity in self.roundboundaries:
                pass
            else:
                self.roundboundaries.append(entity)
                if self.VERBOSE: print(entity.name+" added to roundboundaries")
        if isinstance(entity,Agent):
            if entity in self.agents:
                pass
            else:
                self.agents.append(entity)
                if self.VERBOSE: print(entity.name+" added to agents")
        if isinstance(entity,InteractiveAgent):
            if entity in self.interactiveagents:
                pass
            else:
                self.interactiveagents.append(entity)
                if self.VERBOSE: print(entity.name+" added to interactiveagents")
        if isinstance(entity,Pollen):
            if entity in self.pollens:
                pass
            else:
                self.pollens.append(entity)
                if self.VERBOSE: print(entity.name+" added to pollens")


    def remove_entity(self, entity):
        if isinstance(entity,Pollen):
                self.pollens.remove(entity)
        if isinstance(entity,InteractiveAgent):
                self.interactiveagents.remove(entity)
        if isinstance(entity,Agent):
                self.agents.remove(entity)
        if isinstance(entity,RoundBoundary):
                self.roundboundaries.remove(entity)
        self.entities.remove(entity)
        del entity
    
    def respawn_entity(self, entity):
        new_position = (random.randint(0,self.world.size[0]),random.randint(0,self.world.size[1]))
        entity.set_position(new_position)
        
    def agent_die(self,deadagent):
        # return energy to environment
        energy=deadagent.DefaultEnergy
        while energy > 0.:
            pollen = Pollen(self ,  (random.randint(0,self.world.size[0]),random.randint(0,self.world.size[1])), "pollenflake" )
            energy -= pollen.energy
            self.add_entity(pollen)
        self.remove_entity(deadagent)


    def agent_born(self,parentagent):
        if self.printreproductiveAgent:  self.print_agent_to_file(parentagent)
        if self.agentstatFile is not None: self.printagent(parentagent)
        newagent = parentagent.reproduce(self)
        self.add_entity(newagent)
        self.respawn_entity(newagent)

    def print_agent_to_file(self,agent):
        file1 = open(self.bestagentfilename, "w")
        agent.print_to_file(file1)
        file1.close()


#    def agent_die(self,deadagent):
#        # search for best agent
#        max_energy = -1.
#        bestAgent = None
#        for tmpagent in self.agents:
#            if max_energy < tmpagent.get_energy(): 
#                max_energy = tmpagent.get_energy()
#                bestAgent = tmpagent
#        # reproduce best agent
#        newagent = bestAgent.reproduce(self)
#        newagent.set_energy(self.median_agent_energy())
#        self.add_entity(newagent)
#        self.respawn_entity(newagent)
#        # remove dead agent
#        self.remove_entity(deadagent)


#    def decimation(self):
#        self.decimation_number +=1
#        q=0.25
#        upper_quantile = self.agent_energy_quantile(1-q)
#        lower_quantile = self.agent_energy_quantile(q)
#        agentstobeadded = []
#        agentstoberemoved = []
#        for agent in self.agents:
#            if upper_quantile <= agent.get_energy():
#                if self.agentstatFile is not None: self.printagent(agent)
#                newagent = agent.reproduce(self)
#                agentstobeadded.append(newagent)
#            elif lower_quantile < agent.get_energy():
#                agentstoberemoved.append(agent)
#        for oldagent in agentstoberemoved:
#            self.remove_entity(oldagent)
#        for newagent in agentstobeadded:
#            self.add_entity(newagent)
#            self.respawn_entity(newagent)                      
#            newagent.set_energy(lower_quantile) 

    def printagent(self,agent):
        if self.agentstatFile is not None:
                L = [agent.name + ' {} '.format(self.decimation_number) + ' '.join(map(str, agent.get_energyTotInOut())) +  ' {}'.format(agent.get_energy()) + '\n' ]
                self.agentstatFile.writelines(L)
        

#    def agent_die(self,deadagent):
#        self.decimation_number +=1
#        median_energy = self.median_agent_energy()
#        agentstobeadded = []
#        agentstoberemoved = []
#        for tmpagent in self.agents:
#            if self.agentstatFile is not None:
#                L = [tmpagent.name + ' {} '.format(self.decimation_number) + ' '.join(map(str, tmpagent.get_energyTotInOut())) + '\n' ]
#                self.agentstatFile.writelines(L)
#            if median_energy <= tmpagent.get_energy():
#                newagent = tmpagent.reproduce(self)
#                agentstobeadded.append(newagent)
#            else:
#                agentstoberemoved.append(tmpagent)
#        for oldagent in agentstoberemoved:
#            self.remove_entity(oldagent)
#        for newagent in agentstobeadded:
#            self.add_entity(newagent)
#            self.respawn_entity(newagent)                      
#            newagent.set_default_energy() 

    def print_best_agent(self):
        # search for best agent
        max_energy = -1.
        bestAgent = None
        for agent in self.agents:
            agent.turnvisibility(False)
            if max_energy < agent.get_energy(): 
                max_energy = agent.get_energy()
                bestAgent = agent
#        bestAgent.print_brain(file1=None)
#        bestAgent.print_agent(file1=None)
        bestAgent.turnvisibility(True)

    def pollen_eaten(self,pollen):
        self.respawn_entity(pollen)

    def update(self):
        # steps
        now = datetime.now()
        #if now >= self.next_step_time:
        for agent in self.agents:
            agent.update(self)
        self.next_step_time = now + self.TIME_STEP
        self.current_step += 1

        # draws all world objects
        self.world.draw(self, self.entities)

    def median_agent_energy(self):
        energies = np.zeros(len(self.agents))
        for k,agent in enumerate(self.agents):
            energies[k] = agent.get_energy()
        median = np.median(energies,overwrite_input=True)
        print(energies)
        print("median={}".format(median))
        return median

    def total_agent_energy(self):
        energy = 0. 
        for agent in self.agents:
            energy += agent.get_energy()
        return energy


    def agent_energy_quantile(self,q):
        energies = np.zeros(len(self.agents))
        for k,agent in enumerate(self.agents):
            energies[k] = agent.get_energy()
        return  np.quantile(energies,q,overwrite_input=True)


    def average_agent_energy(self):
        average_energy = 0.
        for agent in self.agents:
            average_energy += agent.get_energy()
        if (len(self.agents)>0):
            average_energy = average_energy / len(self.agents)
        return average_energy

    def n_agent(self):
        return len(self.agents)

    def print_agents(self):
        for agent in self.agents:
            if isinstance(agent,IntelligentAgent):
                agent.print_brain()
        
    def process_event(self,event):
        for agent in self.interactiveagents:
            agent.process_event(event)


class CambrianZoo(Simulation):

    def populate_zoo(self):
        print("create simulation")
        #### POLLEN
        for i in range(1,180):
            pollen = Pollen(self ,  (random.randint(100, 1000),random.randint(100, 1000)), "pollenflake{}".format(i) )       
            self.add_entity(pollen)
        #### ROUNDBOUNDARY
        for i in range(10,20):
            self.add_entity(RoundBoundary(self , (random.randint(100, 1000),random.randint(100, 1000)) , random.randint(10,30), "b{}".format(i) )) 
        #### NEMO
        nemo = NemoFish(self , (500,300) ,"nemo1" )       
        self.add_entity(nemo)
        self.target = nemo
        #### SHARKS
        for i in range(10,18):
                    shark = Follower2(self , (random.randint(100, 1000),random.randint(100, 1000)) ,"shark{}".format(i) ) 
                    self.add_entity(shark)


class EdiacaranZoo(Simulation):

    intelligent_agent_surface_density = 1.0e-5
    pollen_surface_density = 1.0e-4 
    pollen_threshold = 0.9

    def populate_zoo(self):
        print("create simulation")
        #### POLLEN
        for i in range(1, round(self.pollen_surface_density * self.world.size[0] * self.world.size[1])   ):
            pollen = Pollen(self ,  (random.randint(0,self.world.size[0]),random.randint(0,self.world.size[1])), "pollenflake{}".format(i) )       
            self.add_entity(pollen)
        #### NEMO
        #nemo = NemoFish(self , (500,300) ,"nemo1" )       
        for i in range(10,11 + round(self.intelligent_agent_surface_density * self.world.size[0] * self.world.size[1]) ):
            nemo = annAgent(self ,  (random.randint(0,self.world.size[0]),random.randint(0,self.world.size[1])), "annNemo{}_".format(i) )       
            self.add_entity(nemo)
        #self.target = nemo

        for i in range(10,11 + round(self.intelligent_agent_surface_density * self.world.size[0] * self.world.size[1]) ):
            nemo = rnnAgent(self ,  (random.randint(0,self.world.size[0]),random.randint(0,self.world.size[1])), "rnnNemo{}_".format(i) )       
            self.add_entity(nemo)
        #self.target = nemo
        self.embodied_energy = self.total_agent_energy()
   
    def pollen_eaten(self,pollen):
        if self.total_agent_energy() < self.embodied_energy :
            self.respawn_entity(pollen)
        else:
            self.remove_entity(pollen)


    def process_event(self,event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_PAGEUP: 
            self.embodied_energy += 0.05
        if event.type == pygame.KEYDOWN and event.key == pygame.K_PAGEDOWN:
            self.embodied_energy -= 0.05
        super().process_event(event)

 















