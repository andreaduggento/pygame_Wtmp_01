from __future__ import annotations
import random
from datetime import datetime, timedelta
from main.entities.Agents import *
from main.entities.Pollen import Pollen
from main.entities.RoundBoundary import RoundBoundary
from main.entities.NemoFish import *
from main.entities.Sharks import *
from main.world.World import World, distance_between

class SimulationFlock:

    SPOT_TIME_STEP = timedelta(seconds=1000.0)
    DeltaT = 0.01
    TIME_STEP = timedelta(seconds=DeltaT)

    FRICTION = 1.

    def __init__(self,worldsize):
        self.size=worldsize
        self.entities = []
        self.roundboundaries = []
        self.agents = []
        self.interactiveagents = []
        self.world = World(worldsize)
        self.create_simulation()
        self.next_spot_time = datetime.now()
        self.next_step_time = datetime.now()
        self.worldtorus = True
        self.current_step = 1
        ''' # statistics '''
#        self.persons_solved = 0
#        self.spots_solved = 0

    def create_simulation(self):
        print("create simulation")

        #### ROUNDBOUNDARY
        for i in range(10,20):
            self.add_entity(RoundBoundary(self , (random.randint(100, 1000),random.randint(100, 1000)) , random.randint(10,30), "b{}".format(i) ))       
        #### NEMO
        nemo = NemoFish(self , (500,300) ,"nemo1" )       
        self.add_entity(nemo)
        self.target = nemo
        #### SHARKS
        for i in range(10,20):
                    shark = Flocker1(self , (300, 50+ i*50) ,"shark{}".format(i) ) 
                    self.add_entity(shark)

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
                print(entity.name+" added to roundboundaries")
        if isinstance(entity,Agent):
            if entity in self.agents:
                pass
            else:
                self.agents.append(entity)
                print(entity.name+" added to agents")
        if isinstance(entity,InteractiveAgent):
            if entity in self.interactiveagents:
                pass
            else:
                self.interactiveagents.append(entity)
                print(entity.name+" added to interactiveagents")


    def remove_entity(self, entity):
        self.entities.remove(entity)

    def update(self):
        # steps
        now = datetime.now()
        if now >= self.next_step_time:
            for agent in self.agents:
                agent.update(self)
            self.next_step_time = now + self.TIME_STEP
            self.current_step += 1
        # draws all world objects
        self.world.draw(self, self.entities)

    def process_event(self,event):
            for agent in self.interactiveagents:
                agent.process_event(event)




class Simulation:

    SPOT_TIME_STEP = timedelta(seconds=1000.0)
    DeltaT = 0.01
    TIME_STEP = timedelta(seconds=DeltaT)

    FRICTION = 1.

    def __init__(self,worldsize):
        self.size=worldsize
        self.entities = []
        self.roundboundaries = []
        self.agents = []
        self.interactiveagents = []
        self.world = World(worldsize)
        self.create_simulation()
        self.next_spot_time = datetime.now()
        self.next_step_time = datetime.now()
        self.worldtorus = True
        self.current_step = 1
        ''' # statistics '''
#        self.persons_solved = 0
#        self.spots_solved = 0

    def create_simulation(self):
        print("create simulation")

        #### POLLEN
        pollen = Pollen(self , (100,100) ,"pollenflake" )       
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
                print(entity.name+" added to roundboundaries")
        if isinstance(entity,Agent):
            if entity in self.agents:
                pass
            else:
                self.agents.append(entity)
                print(entity.name+" added to agents")
        if isinstance(entity,InteractiveAgent):
            if entity in self.interactiveagents:
                pass
            else:
                self.interactiveagents.append(entity)
                print(entity.name+" added to interactiveagents")


    def remove_entity(self, entity):
        self.entities.remove(entity)

    def update(self):
        # steps
        now = datetime.now()
        if now >= self.next_step_time:
            for agent in self.agents:
                agent.update(self)
            self.next_step_time = now + self.TIME_STEP
            self.current_step += 1
        # draws all world objects
        self.world.draw(self, self.entities)

    def process_event(self,event):
            for agent in self.interactiveagents:
                agent.process_event(event)




















