#!/usr/bin/env python
# Author: Mardcore

from queue import PriorityQueue
import math
import time
import itertools
from abc import ABC, abstractmethod

import geopandas as gpd
from shapely.geometry import Point

class Action:
    def __init__(self, origin, dest, cost):
        self.origin=origin
        self.destination=dest
        self.cost=cost

    def __repr__(self):
        return 'Action('+ str(self.origin) + ', ' + str(self.destination) + ', ' + str(self.cost) + ')'
    
    def __str__(self):
        return 'origin city: ' + str(self.origin) + ', destination city: ' + str(self.destination) + ', cost: ' + str(self.cost)
    
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.origin == other.origin and self.destination == other.destination and self.cost == other.cost

class State:
    def __init__(self, id):
        self.id=id

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.id == other.id
        
    def __hash__(self):
        return hash(self.id)
    
    def apply_action(self, action):
        return State(action.destination)

    def __repr__(self):
        return 'State('+ str(self.id) + ')'
    
    def __str__(self):
        return 'city: ' + str(self.id)

class Node:
    def __init__(self, state, action, parent, depth):
        self.state=state
        self.action=action
        self.parent=parent
        self.depth=depth
    
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.state == other.state and self.action == other.action and self.parent == other.parent and self.depth == other.depth
       
    def __repr__(self):
        return 'Node('+ str(self.state) + ', ' + str(self.action) + ', ' + str(self.parent) + str(self.depth) + ')'
    
    def __str__(self):
        return 'Actual node ' + str(self.state) + ', action ' + str(self.action) + ', parent node ' + str(self.parent) + ', depth: ' + str(self.depth)

class Problem:

    def __init__(self, problem):

        self.cities = {city['id']: city for city in problem['map']['cities']}
        self.gdf = gpd.GeoDataFrame(problem['map']['cities'])
        self.gdf['Coordinates'] = list(zip(self.gdf.lon, self.gdf.lat))
        self.gdf['Coordinates'] = self.gdf['Coordinates'].apply(Point)
        self.gdf.set_geometry('Coordinates', inplace=True)
        self.map = problem['map']
        self.initial_state = None
        self.final_city = None
        self.actions = self.action()

    def setDepartureGoal(self, departure, goal):
        self.initial_state = State(departure)
        self.final_city = State(goal)

    def action(self):
        actions = {}
        for road in self.map['roads']:
            if road["origin"] not in actions:
                actions[road["origin"]]=[]
            actions[road["origin"]].append(Action(road["origin"], road["destination"], road["distance"]))
        return actions

    def state(self, state):
        return state.id == self.final_city.id

    def nodeactions(self, node):
        return self.actions[node.state.id]

    def plot_map(self, action_list=None, world_name='Spain'):
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        city_ids = {self.cities[city]['name']: city for city in self.cities}
        # We restrict to Spain.
        ax = world[world.name == world_name].plot(
            color='white', edgecolor='black',linewidth=3,figsize=(100,70))

        self.gdf.plot(ax=ax, color='red',markersize=500)
        for x, y, label in zip(self.gdf.Coordinates.x, self.gdf.Coordinates.y, self.gdf.name):
            ax.annotate(f'{city_ids[label]} -- {label}', xy=(x, y), xytext=(8, 3), textcoords="offset points",fontsize=60)
        roads = itertools.chain.from_iterable(self.actions.values())

        for road in roads:
            slat = self.cities[road.origin]['lat']
            slon = self.cities[road.origin]['lon']
            dlat = self.cities[road.destination]['lat']
            dlon = self.cities[road.destination]['lon']
            if action_list and road in action_list:
                color = 'red'
                linewidth = 15
            else:
                color = 'lime'
                linewidth = 5
            ax.plot([slon , dlon], [slat, dlat], linewidth=linewidth, color=color, alpha=0.5)

class Search(ABC):
    def __init__(self, problem):
        self.problem = problem
        self.open_list = []
        self.closed_list = set()
        self.execution_time = 0
        self.solution_cost = 0
        self.solution_depth = 0
        self.nodes_generated = 1
        self.nodes_expanded = 0
        self.max_nodes_in_memory = 0
        self.solution_actions = []
    
    @abstractmethod
    def insert_node(self, node, node_list):
        pass
    
    @abstractmethod
    def extract_node(self, node_list):
        pass

    @abstractmethod
    def is_empty(self, node_list):
        pass
    
    def get_successors(self, node):
        successors = []
        for action in self.problem.nodeactions(node):
            new_state = node.state.apply_action(action)
            new_node = Node(new_state, action, node, node.depth+1)
            successors.append(new_node)
        return successors
    
    def do_search(self):
        start_time = time.perf_counter()
        initial_node = Node(self.problem.initial_state, None, None, 0)
        self.insert_node(initial_node, self.open_list)
        while not self.is_empty(self.open_list):
            node = self.extract_node(self.open_list)
            if self.problem.state(node.state):
                self.solution_depth = node.depth
                self.solution_actions = self.get_solution_path(node)
                self.execution_time = time.perf_counter() - start_time
                return self.solution_actions
            self.closed_list.add(node.state)
            self.nodes_expanded += 1
            successors = self.get_successors(node)
            for successor in successors:
                if successor.state not in self.closed_list:
                    self.insert_node(successor, self.open_list)
                    self.nodes_generated += 1
            if isinstance(self.open_list, PriorityQueue):
                self.max_nodes_in_memory = max(self.max_nodes_in_memory, self.open_list.qsize())
            else:
                self.max_nodes_in_memory = max(self.max_nodes_in_memory, len(self.open_list))
        self.execution_time = time.perf_counter() - start_time
        return False

class DepthFirst(Search):
    def __init__(self, problem):
        super().__init__(problem)
    
    def insert_node(self, node, node_list):
        node_list.append(node)
    
    def extract_node(self, node_list):
        return node_list.pop()
    
    def is_empty(self, node_list):
        return len(node_list) == 0
    
    def get_solution_path(self, node):
        solution_path = []
        while node.parent:
            solution_path.append(node.action)
            self.solution_cost += node.action.cost
            node = node.parent
        return solution_path[::-1]
    

class BreadthFirst(Search):
    def __init__(self, problem):
        super().__init__(problem)
    
    def insert_node(self, node, node_list):
        node_list.append(node)
    
    def extract_node(self, node_list):
        return node_list.pop(0)
    
    def is_empty(self, node_list):
        return len(node_list) == 0

    def get_solution_path(self, node):
        solution_path = []
        while node.parent:
            solution_path.append(node.action)
            self.solution_cost += node.action.cost
            node = node.parent
        return solution_path[::-1]

class BestFirst(Search):
    def __init__(self, parent_args, child_args):
        super().__init__(parent_args)

        self.child_args = child_args
        self.open_list = PriorityQueue()
        self.counter1 = 0

    def insert_node(self, node, node_list:PriorityQueue):
        self.counter1 += 1
        return node_list.put((self.child_args.get_hcost(node),self.counter1, node))
        
    def extract_node(self, node_list:PriorityQueue):
        return node_list.get()[2]

    def is_empty(self, node_list:PriorityQueue):
        return node_list.qsize() == 0

    def get_solution_path(self, node):
        solution_path = []
        while node.parent:
            solution_path.append(node.action)
            self.solution_cost += node.action.cost
            node = node.parent
        return solution_path[::-1]

class AStar(Search):
    def __init__(self, parent_args, child_args):
        
        super().__init__(parent_args)

        self.child_args = child_args
        self.open_list = PriorityQueue()
        self.counter2 = 0

    def insert_node(self, node, node_list:PriorityQueue):
        self.counter2 += 1
        return node_list.put((self.child_args.get_hcost(node)+self.child_args.get_gcost(node),self.counter2, node))
        
    def extract_node(self, node_list:PriorityQueue):
        return node_list.get()[2]

    def is_empty(self, node_list:PriorityQueue):
        return node_list.qsize() == 0

    def get_solution_path(self, node):
        solution_path = []
        while node.parent:
            solution_path.append(node.action)
            self.solution_cost += node.action.cost
            node = node.parent
        return solution_path[::-1]

class Heuristic(ABC):   
    @abstractmethod
    def get_hcost(self, node):
        pass

class BestFirstHeuristic(Heuristic):
    def __init__(self, info):
        self.info = info

    def get_hcost(self, node):
        rad=math.pi/180
        dlat=self.info.map['cities'][self.info.final_city.id]['lat'] - self.info.map['cities'][node.state.id]['lat']
        dlon=self.info.map['cities'][self.info.final_city.id]['lon'] - self.info.map['cities'][node.state.id]['lon']
        r=6372.795477598
        a=(math.sin(rad*dlat/2))**2 + math.cos(rad*self.info.map['cities'][node.state.id]['lat'])*math.cos(rad*self.info.map['cities'][self.info.final_city.id]['lat'])*(math.sin(rad*dlon/2))**2
        distance=2*r*math.asin(math.sqrt(a))
        return distance

class AStarHeuristic(Heuristic):
    def __init__(self, info):
        self.info = info

    def get_hcost(self, node):
        rad=math.pi/180
        dlat=self.info.map['cities'][self.info.final_city.id]['lat'] - self.info.map['cities'][node.state.id]['lat']
        dlon=self.info.map['cities'][self.info.final_city.id]['lon'] - self.info.map['cities'][node.state.id]['lon']
        r=6372.795477598
        a=(math.sin(rad*dlat/2))**2 + math.cos(rad*self.info.map['cities'][node.state.id]['lat'])*math.cos(rad*self.info.map['cities'][self.info.final_city.id]['lat'])*(math.sin(rad*dlon/2))**2
        distance=2*r*math.asin(math.sqrt(a))
        return distance

    def get_gcost(self, node):
        accumulated_cost = 0
        while node.parent:
            accumulated_cost += node.action.cost
            node = node.parent
        return accumulated_cost