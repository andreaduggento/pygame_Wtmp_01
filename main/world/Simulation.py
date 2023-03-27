from __future__ import annotations
import random
from datetime import datetime, timedelta
from main.entities.Agents import *
from main.entities.Pollen import Pollen
from main.entities.RoundBoundary import RoundBoundary
from main.entities.NemoFish import *
from main.entities.Sharks import *
from main.world.World import World, distance_between

class Simulation:

    SPOT_TIME_STEP = timedelta(seconds=1000.0)
    DeltaT = 0.02
    TIME_STEP = timedelta(seconds=DeltaT)
    VERBOSE = False
    FRICTION = 1.

    def __init__(self,worldsize):
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
#        self.persons_solved = 0
#        self.spots_solved = 0

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
        # search for best agent
        max_energy = -1.
        bestAgent = None
        for tmpagent in self.agents:
            if max_energy < tmpagent.get_energy(): 
                max_energy = tmpagent.get_energy()
                bestAgent = tmpagent
        # reproduce best agent
        newagent = bestAgent.reproduce(self)
        self.add_entity(newagent)
        self.respawn_entity(newagent)
        # remove dead agent
        self.remove_entity(deadagent)

    def print_best_agent(self):
        # search for best agent
        max_energy = -1.
        bestAgent = None
        for agent in self.agents:
            agent.turnvisibility(False)
            if max_energy < agent.get_energy(): 
                max_energy = agent.get_energy()
                bestAgent = agent
        bestAgent.print_brain()
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

    def __init__(self,worldsize):
        super().__init__(worldsize)

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

    intelligent_agent_surface_density = .000006
    pollen_surface_density = .00008


    def __init__(self,worldsize):
        super().__init__(worldsize)

    def populate_zoo(self):
        print("create simulation")
        #### POLLEN
        for i in range(1, round(self.pollen_surface_density * self.world.size[0] * self.world.size[1])   ):
            pollen = Pollen(self ,  (random.randint(0,self.world.size[0]),random.randint(0,self.world.size[1])), "pollenflake{}".format(i) )       
            self.add_entity(pollen)
        #### NEMO
        #nemo = NemoFish(self , (500,300) ,"nemo1" )       
        for i in range(1, round(self.intelligent_agent_surface_density * self.world.size[0] * self.world.size[1]) ):
            nemo = annAgent(self ,  (random.randint(0,self.world.size[0]),random.randint(0,self.world.size[1])), "nemo{}".format(i) )       
            self.add_entity(nemo)
        #self.target = nemo

        for i in range(1, round(self.intelligent_agent_surface_density * self.world.size[0] * self.world.size[1]) ):
            nemo = rnnAgent(self ,  (random.randint(0,self.world.size[0]),random.randint(0,self.world.size[1])), "nemo{}".format(i) )       
            self.add_entity(nemo)
        #self.target = nemo
   
    def pollen_eaten(self,pollen):
        if self.average_agent_energy() < .9 :
            self.respawn_entity(pollen)
        else:
            self.remove_entity(pollen)

 















