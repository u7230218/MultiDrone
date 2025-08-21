import numpy as np
from multi_drone import MultiDrone

# Initialize the MultiDrone environment
sim = MultiDrone(num_drones=2, environment_file="environment.yaml")

# Obtain the initial configuration and the goal positions
initial_configuration = sim.initial_configuration
goal_positions = sim.goal_positions

class Graph_Node:
    def __init__(self, state, path_cost=0, parent=None, depth=0):
        self.state = state
        self.path_cost = path_cost
        self.parent = parent
        self.depth = depth

# Samples a configuration around the prev_point by a fixed distance
def sample_configuration(prev_point, distance):
    lower_bound = prev_point - distance
    Upper_bound = prev_point + distance
    return np.random.uniform(low= lower_bound, high=Upper_bound, size=prev_point.size)

# Finding path from start point to end point within the graph
def search_graph(sim, graph, start, end):
    # explored = set()
    # initial_node = Graph_Node(start)
    # frontier = [initial_node]
    # while frontier:
    #     node = frontier.pop()
    #     if sim.
    return None

# Creates plan while only considering the presence of a single dron
def single_drone_planner(sim, drone_configs, total_iterations=10000):
    # Obtaining the initial configuration and goal position for the drone
    initial_config, goal_config = drone_configs
    # Checking if initial configuration to goal position is a valid path
    if sim.motion_valid(initial_config, goal_config):
        return [initial_config, goal_config]
    # Creating roadmap (Interleave)
    distance = np.linalg.norm(initial_config, goal_config)
    cur_iteration = 0
    road_map = {initial_config: [], goal_config: []}
    prev_config = initial_config
    while cur_iteration != total_iterations:
        plan = search_graph(sim, road_map, initial_config, goal_config)
        if plan != None:
            return plan
        sampled_config = sample_configuration(prev_config, distance)
        if sim.is_valid(sampled_config) and sampled_config not in road_map.keys():
            road_map[sampled_config] = set()
            prev_config = sampled_config
            for key in road_map.keys():
                if sim.motion_valid(key, sampled_config):
                    edge_weight = np.linalg.norm(initial_config, goal_config)
                    road_map[sampled_config].append((key, edge_weight))
                    road_map[key].append((sampled_config, edge_weight))
        cur_iteration += 1
    print(" Could not find path in ", total_iterations, " iterations")
    return None