import numpy as np
from multi_drone import MultiDrone

# Class that defines the nodes within the graph
class Graph_Node:
    def __init__(self, state, cost=None, parent=None, depth=0):
        self.state = state
        if cost is None:
            self.cost = np.zeros((state.shape[0]))
        else:
            self.cost = cost
        self.parent = parent
        self.depth = depth

# Queue used in ordering of the frontier in graph search
class Graph_Queue:
    def __init__(self):
        self.elements = {}
        self._min_priority = None
        self._sorted_priority_list = []
    
    def push(self, graph_node, priority):
        # Adding node of a given priority to queue
        if priority not in self.elements.keys():
            self.elements[priority] = [graph_node]
        else:
            self.elements[priority].append(graph_node)

        # Replacing the current min priority in Queue
        if self._min_priority == None or priority < self._min_priority:
            self._min_priority = priority

        self._sorted_priority_list.append(priority)
        self._sorted_priority_list.sort(reverse=True)
    
    def pop(self):
        if self._min_priority != None:
            min_node = self.elements[self._min_priority].pop()
            if len(self.elements[self._min_priority]) == 0: # Update the value of the min priority
                # Remove dictionary enrty
                self.elements.pop(self._min_priority)
                # Update the sorted priority list
                self._sorted_priority_list.pop()
                if len(self._sorted_priority_list) == 0:
                    self._min_priority = None
                else:
                    self._min_priority = self._sorted_priority_list[-1]
            return min_node
        return None

# Expanding successor nodes for a given node in a graph
def expand_node(graph, node):
    successor_nodes = []
    for succ_state, cost in graph[tuple(map(tuple, node.state))]:
        successor_nodes.append(Graph_Node(succ_state, cost=cost + node.cost, parent=node, depth=node.depth + 1))
    return successor_nodes

# Heuristic used in the graph search (straight line distance to goal state)
def graph_heuristic(state, goal_state):
    return np.linalg.norm(state - goal_state, axis=1)

# Adds the cost of the node and its heuristic
def graph_function(node, goal_state):
    return np.sum(node.cost + graph_heuristic(node.state, goal_state))

# Finding path from start point to end point within the graph
def search_graph(sim, graph, start, goal):
    explored = set()
    initial_node = Graph_Node(start)
    frontier = Graph_Queue()
    frontier.push(initial_node, graph_function(initial_node, goal))
    observed_nodes = {}
    observed_nodes[tuple(map(tuple, start))] = initial_node
    while frontier.elements:
        cur_node = frontier.pop()
        if sim.is_goal(cur_node.state):
            return cur_node
        explored.add(tuple(map(tuple, cur_node.state)))
        for child_node in expand_node(graph, cur_node):
            m_node = None
            if tuple(map(tuple, child_node.state)) in observed_nodes:
                m_node = observed_nodes[tuple(map(tuple, child_node.state))]
            if m_node == None: # there exists no node in frontier or explored such that it shares a state with the child node
                frontier.push(child_node, graph_function(child_node, goal))
                observed_nodes[tuple(map(tuple, child_node.state))] = child_node
            elif (child_node.cost < m_node.cost).all():
                m_node.cost = child_node.cost
                m_node.parent = child_node.parent
                m_node.depth = child_node.depth
                if tuple(map(tuple, m_node.state)) in explored:
                    explored.remove(tuple(map(tuple, m_node.state)))
                    frontier.push(m_node, graph_function(m_node, goal))
    return None

# Returns sequence of waypoints
def find_path_to_goal(sim, graph, start, end):
    leaf_node = search_graph(sim, graph, start, end)
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
def my_planner(sim, total_iterations=1000):
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
        plan = find_path_to_goal(sim, road_map, initial_config, goal_config)
        if plan != None:
            return plan
        sampled_config = sample_configuration(prev_config, distance)
        if sim.is_valid(sampled_config) and tuple(map(tuple, sampled_config)) not in road_map.keys():
            road_map[tuple(map(tuple, sampled_config))] = []
            prev_config = sampled_config
            for key in road_map.keys():
                if sim.motion_valid(np.array(key), sampled_config):
                    edge_weight = np.linalg.norm(np.array(key) - sampled_config, axis=1)
                    road_map[tuple(map(tuple, sampled_config))].append((np.array(key), edge_weight))
                    road_map[key].append((sampled_config, edge_weight))
        cur_iteration += 1
        print("cur_iteration = ", cur_iteration)
    print(" Could not find path in ", total_iterations, " iterations")
    return None

# Initialize the MultiDrone environment
sim = MultiDrone(num_drones=3, environment_file="test_drone_3.yaml")

paths = my_planner(sim)
sim.visualize_paths(paths)