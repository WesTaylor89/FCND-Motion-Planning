import networkx as nx
import numpy as np
from bresenham import bresenham
from shapely.geometry import Point


def prune_path_bresenham(path):
    """ Prune the path using Bresenham's ray tracing algorithm. """
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
    """ Calculate the distance of a point from a line defined by start and end points. """
    if np.all(np.equal(start, end)):
        return np.linalg.norm(point - start)

    return np.divide(
        np.abs(np.linalg.norm(np.cross(end - start, start - point))),
        np.linalg.norm(end - start))


def prune_path_collinearity(path, epsilon=1e-5):
    """ Prune the path using collinearity check. """
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
    # print("Path pruned: ", pruned_path_combined)
    return pruned_path_combined


def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))


def a_star_graph(graph, start, goal):
    path = nx.astar_path(graph, start, goal, heuristic=heuristic)
    return path


def closest_point(graph, point):
    """ Find the closest point in the graph to the given point. """
    closest_point = None
    min_dist = np.inf
    for p in graph.nodes:
        dist = np.linalg.norm(np.array(p) - np.array(point))
        if dist < min_dist:
            closest_point = p
            min_dist = dist
    return closest_point


def collides(index_lock, polygons_index, point, polygons):
    point_obj = Point(point[0], point[1])
    with index_lock:
        for idx in polygons_index.intersection((point[0], point[1], point[0], point[1])):
            poly, height = polygons[idx]
            if poly.contains(point_obj) and height > point[2]:
                return True
    return False


def get_bounds(data):
    """ Find bounds for the operational area based on obstacle data. """
    xmin = np.min(data[:, 0] - data[:, 3])
    xmax = np.max(data[:, 0] + data[:, 3])
    ymin = np.min(data[:, 1] - data[:, 4])
    ymax = np.max(data[:, 1] + data[:, 4])
    zmin = 0
    zmax = 100  # Example z bounds
    return xmin, xmax, ymin, ymax, zmin, zmax


def calculate_heading(wp1, wp2):
    # Calculate the heading angle between two waypoints
    heading = np.arctan2((wp2[1] - wp1[1]), (wp2[0] - wp1[0]))
    return heading
