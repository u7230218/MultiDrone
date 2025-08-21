import numpy as np
from multi_drone import MultiDrone

# Initialize the MultiDrone environment
sim = MultiDrone(num_drones=2, environment_file="environment.yaml")

class Graph_Node:
    def __init__(self, state, cost=None, parent=None, depth=0):
        self.state = state
        if cost is None:
            self.cost = np.zeros((state.shape[0]))
        else:
            self.cost = cost
        self.parent = parent
        self.depth = depth

# Expanding successor nodes for a given node in a graph
def expand_node(graph, node):
    successor_nodes = [] # TODO: Replace with a priority queue
    for succ_state, cost in graph[tuple(map(tuple, node.state))]:
        successor_nodes.append(Graph_Node(succ_state, cost=cost + node.cost, parent=node, depth=node.depth + 1))
    return successor_nodes

# Finding path from start point to end point within the graph
def search_graph(sim, graph, start):
    explored = set()
    initial_node = Graph_Node(start)
    frontier = [initial_node] # TODO: Replace with a priority queue
    observed_nodes = {}
    observed_nodes[tuple(map(tuple, start))] = initial_node
    while frontier:
        cur_node = frontier.pop()
        if sim.is_goal(cur_node.state):
            return cur_node
        explored.add(tuple(map(tuple, cur_node.state)))
        for child_node in expand_node(graph, cur_node):
            m_node = None
            if tuple(map(tuple, child_node.state)) in observed_nodes:
                m_node = observed_nodes[tuple(map(tuple, child_node.state))]
            if m_node == None: # there exists no node in frontier or explored such that it shares a state with the child node
                frontier.append(child_node)
                observed_nodes[tuple(map(tuple, child_node.state))] = child_node
            elif (child_node.cost < m_node.cost).all():
                m_node.cost = child_node.cost
                m_node.parent = child_node.parent
                m_node.depth = child_node.depth
                if tuple(map(tuple, m_node.state)) in explored:
                    explored.remove(tuple(map(tuple, m_node.state)))
                    frontier.append(m_node)
    return None

# Returns sequence of waypoints
def find_path_to_goal(sim, graph, start):
    leaf_node = search_graph(sim, graph, start)
    if leaf_node == None:
        return None
    solution = []
    cur_node = leaf_node
    while cur_node != None:
        solution.insert(0, cur_node.state)
        cur_node = cur_node.parent
    return solution

# Samples a configuration around the prev_point by a fixed distance
def sample_configuration(prev_point, distance):
    lower_bound = prev_point - distance[:, None]
    Upper_bound = prev_point + distance[:, None]
    return np.random.uniform(low=np.maximum(lower_bound, np.zeros_like(lower_bound)), high=np.minimum(Upper_bound, np.full(Upper_bound.shape, 50, dtype=np.int32)), size=prev_point.shape)

# Creates plan while only considering the presence of a single dron
def my_planner(sim, total_iterations=10000):
    # Obtain the initial configuration and the goal positions
    initial_config = sim.initial_configuration
    goal_config = sim.goal_positions
    # Checking if initial configuration to goal position is a valid path
    if sim.motion_valid(initial_config, goal_config):
        return [initial_config, goal_config]
    # Creating roadmap (Interleave)
    distance = np.linalg.norm(initial_config - goal_config, axis=1)
    cur_iteration = 0
    road_map = {tuple(map(tuple, initial_config)): [], tuple(map(tuple, goal_config)): []}
    prev_config = initial_config
    while cur_iteration != total_iterations:
        plan = find_path_to_goal(sim, road_map, initial_config)
        if plan != None:
            return plan
        sampled_config = sample_configuration(prev_config, distance)
        if sim.is_valid(sampled_config) and tuple(map(tuple, sampled_config)) not in road_map.keys():
            road_map[tuple(map(tuple, sampled_config))] = []
            prev_config = sampled_config
            for key in road_map.keys():
                if sim.motion_valid(np.array(key), sampled_config):
                    edge_weight = np.linalg.norm(initial_config - goal_config, axis=1)
                    road_map[tuple(map(tuple, sampled_config))].append((np.array(key), edge_weight))
                    road_map[key].append((sampled_config, edge_weight))
        cur_iteration += 1
        print("cur_iteration = ", cur_iteration)
    print(" Could not find path in ", total_iterations, " iterations")
    return None

paths = my_planner(sim)
sim.visualize_paths(paths)