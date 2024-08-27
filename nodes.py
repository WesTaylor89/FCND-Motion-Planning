import threading
import time

import networkx as nx
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from scipy.spatial import KDTree
from shapely.geometry import LineString, box

from planning_utils import get_bounds, collides


def create_nodes(map_data, num_nodes, rtree, polygons):
    # Get boundaries of the mapped area (defined by the obstacle data)
    xmin, xmax, ymin, ymax, zmin, zmax = get_bounds(map_data)

    # Create list of randomised nodes
    xvals = np.random.uniform(xmin, xmax, num_nodes)
    yvals = np.random.uniform(ymin, ymax, num_nodes)
    zvals = np.random.uniform(zmin, zmax, num_nodes)
    node_list = list(zip(xvals, yvals, zvals))
    # print("Samples Generated")

    # Check each sample node to see if it is in collision, checks are performed in parallel
    index_lock = threading.Lock()
    with ThreadPoolExecutor() as executor:
        collision_results = list(executor.map(lambda p: collides(index_lock, rtree, p, polygons), node_list))

    # Pair each node with its collision result, only retain nodes that do not collide
    nodes_to_keep = [node for node, collides in zip(node_list, collision_results) if not collides]

    return nodes_to_keep


def create_kdTree(nodes):
    # Convert the list to_keep to a NumPy array
    nodes_nparray = np.array(nodes)

    # Build the KDTree with obstacles in free space
    node_kdTree = KDTree(nodes_nparray)
    # print("KDTree Created")
    return node_kdTree


def create_graph_parallel(to_keep, radius, rtree, polygons):
    """ Creates a graph by connecting points that can be linked without intersecting obstacles. """
    start_time = time.time()

    def can_connect_two_points(p1, p2, polygons):
        line = LineString([p1, p2])
        line_bbox = box(min(p1[0], p2[0]), min(p1[1], p2[1]), max(p1[0], p2[0]), max(p1[1], p2[1]))

        for poly, _ in polygons:
            poly_bbox = poly.bounds
            poly_bbox = box(poly_bbox[0], poly_bbox[1], poly_bbox[2], poly_bbox[3])

            if line_bbox.intersects(poly_bbox):
                if line.intersects(poly):
                    return False
        return True

    def process_point(index):
        point = to_keep[index]
        neighbors = rtree.query_ball_point(point, r=radius)
        edges = []
        for neighbor_index in neighbors:
            neighbor = to_keep[neighbor_index]
            if point != neighbor and can_connect_two_points(point, neighbor, polygons):
                edges.append((point, neighbor))
        return edges

    G = nx.Graph()
    for point in to_keep:
        G.add_node(point)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_point, range(len(to_keep))))

    for edges in results:
        G.add_edges_from(edges)

    end_time = time.time()
    print(f"create_graph_parallel executed in {end_time - start_time:.2f} seconds")
    return G


def create_nodes_and_graph(map_data, num_nodes, rtree, polygons, radius):
    print("Creating nodes and graph")
    nodes = create_nodes(map_data, num_nodes, rtree, polygons)
    node_kdTree = create_kdTree(nodes)
    graph = create_graph_parallel(nodes, radius, node_kdTree, polygons)
    print("Graph Created")
    return graph
