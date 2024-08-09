from enum import Enum
from queue import PriorityQueue
import numpy as np
from math import sqrt
from bresenham import bresenham
import networkx as nx


def create_grid(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size - 1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size - 1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size - 1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size - 1)),
            ]
            grid[obstacle[0]:obstacle[1] + 1, obstacle[2]:obstacle[3] + 1] = 1

    return grid, int(north_min), int(east_min)


# Assume all actions cost the same.
class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """

    WEST = (0, -1, 1)
    EAST = (0, 1, 1)
    NORTH = (-1, 0, 1)
    SOUTH = (1, 0, 1)
    DIAGONAL_UP_RIGHT = (-1, 1, sqrt(2))
    DIAGONAL_UP_LEFT = (-1, -1, sqrt(2))
    DIAGONAL_DOWN_RIGHT = (1, 1, sqrt(2))
    DIAGONAL_DOWN_LEFT = (1, -1, sqrt(2))

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])


def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # check if the node is off the grid or
    # it's an obstacle

    if x - 1 < 0 or grid[x - 1, y] == 1:
        valid_actions.remove(Action.NORTH)
    if x + 1 > n or grid[x + 1, y] == 1:
        valid_actions.remove(Action.SOUTH)
    if y - 1 < 0 or grid[x, y - 1] == 1:
        valid_actions.remove(Action.WEST)
    if y + 1 > m or grid[x, y + 1] == 1:
        valid_actions.remove(Action.EAST)

    # Check for diagonal movements
    if x - 1 < 0 or y + 1 > m or grid[x - 1, y + 1] == 1:
        valid_actions.remove(Action.DIAGONAL_UP_RIGHT)
    if x - 1 < 0 or y - 1 < 0 or grid[x - 1, y - 1] == 1:
        valid_actions.remove(Action.DIAGONAL_UP_LEFT)
    if x + 1 > n or y + 1 > m or grid[x + 1, y + 1] == 1:
        valid_actions.remove(Action.DIAGONAL_DOWN_RIGHT)
    if x + 1 > n or y - 1 < 0 or grid[x + 1, y - 1] == 1:
        valid_actions.remove(Action.DIAGONAL_DOWN_LEFT)

    return valid_actions


def a_star(grid, h, start, goal):
    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:
            current_cost = branch[current_node][0]

        if current_node == goal:
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = current_cost + action.cost
                queue_cost = branch_cost + h(next_node, goal)

                if next_node not in visited:
                    visited.add(next_node)
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))

    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')
    return path[::-1], path_cost


def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))


def prune_path_bresenham(path):
    """
    Prune the path using Bresenham's ray tracing algorithm.
    :param path: List of waypoints (x, y)
    :return: Pruned list of waypoints
    """
    if not path:
        return path

    pruned_path = [path[0]]
    for i in range(1, len(path) - 1):
        p1 = pruned_path[-1]
        p2 = path[i + 1]

        if list(bresenham(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))) != [p1, p2]:
            pruned_path.append(path[i])

    pruned_path.append(path[-1])
    # Ensure the pruned path contains integer coordinates
    pruned_path = [(int(p[0]), int(p[1]), int(p[2])) for p in pruned_path]
    return pruned_path


def point_line_distance(point, start, end):
    """Calculate the distance of a point from a line defined by start and end points."""
    if np.all(np.equal(start, end)):
        return np.linalg.norm(point - start)

    return np.divide(
        np.abs(np.linalg.norm(np.cross(end - start, start - point))),
        np.linalg.norm(end - start))


def prune_path_collinearity(path, epsilon=1e-5):
    """
        Prune the path using collinearity check.
        :param path: List of waypoints (x, y)
        :param epsilon: Minimum distance to consider for pruning
        :return: Pruned list of waypoints
        """
    if len(path) < 3:
        return path

    pruned_path = [path[0]]
    for i in range(1, len(path) - 1):
        p1 = np.array(pruned_path[-1])
        p2 = np.array(path[i])
        p3 = np.array(path[i + 1])

        if point_line_distance(p2, p1, p3) < epsilon:
            continue

        pruned_path.append(path[i])

    pruned_path.append(path[-1])
    # Ensure the pruned path contains integer coordinates
    pruned_path = [(int(p[0]), int(p[1]), int(p[2])) for p in pruned_path]
    return pruned_path


def combined_pruning(path):
    pruned_path_collinearity = prune_path_collinearity(path)
    pruned_path_combined = prune_path_bresenham(pruned_path_collinearity)
    # Ensure the pruned path contains integer coordinates
    pruned_path_combined = [(int(p[0]), int(p[1]), int(p[2])) for p in pruned_path_combined]
    return pruned_path_combined


def a_star_graph(graph, start, goal):
    path = nx.astar_path(graph, start, goal, heuristic=heuristic)
    return path


def closest_point(graph, point):
    """
    Find the closest point in the graph to the given point.
    """
    closest_point = None
    min_dist = np.inf
    for p in graph.nodes:
        dist = np.linalg.norm(np.array(p) - np.array(point))
        if dist < min_dist:
            closest_point = p
            min_dist = dist
    return closest_point
