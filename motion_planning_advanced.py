import argparse
import time
import msgpack
from enum import Enum, auto
import threading

import shapely
from networkx import NetworkXNoPath
from shapely.geometry import Polygon, Point, LineString, box, MultiPolygon
import rtree
from scipy.spatial import KDTree
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from shapely.ops import unary_union

from planning_utils import *
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local


class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):

    def __init__(self, connection, precomputed_polygons=None, precomputed_rtree_index=None, colliders_data=None):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}
        self.polygons = precomputed_polygons if precomputed_polygons is not None else []
        self.rtree_index = precomputed_rtree_index
        self.data = colliders_data

        # initial stateF
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("ARMING TRANSITION")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("TAKEOFF TRANSITION")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("WAYPOINT TRANSITION")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2],
                          self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("LANDING TRANSITION")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("DISARMING TRANSITION")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("Manual TRANSITION")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        print(f"Waypoints data: {data}")
        self.connection._master.write(data)

    def extract_polygons(self, data):
        polygons = []
        for i in range(data.shape[0]):
            # center north, east etc and half width north, east etc
            # of obstacles.
            north, east, alt, d_north, d_east, d_alt = data[i, :]

            # required calculations to calculate corner coordinates
            # from obstacle center.
            obstacle = [
                int(north - d_north),
                int(north + d_north),
                int(east - d_east),
                int(east + d_east)
            ]

            # corners will be:
            # bl = n - dn, e - de, br = n - dn, e + de
            # tr = n + dn, e + de, tl = n + dn, e - de
            corners = [(obstacle[0], obstacle[2]), (obstacle[0], obstacle[3]),
                       (obstacle[1], obstacle[3]), (obstacle[1], obstacle[2])]

            height = int(alt + d_alt)

            p = Polygon(corners)
            polygons.append((p, height))

        return polygons

    def create_polygons_index(self):
        index = rtree.index.Index()
        for i, (poly, height) in enumerate(self.polygons):
            if isinstance(poly, shapely.geometry.MultiPolygon):
                for p in poly:
                    index.insert(i, p.bounds)
            elif isinstance(poly, shapely.geometry.Polygon):
                index.insert(i, poly.bounds)
        return index

    def choose_random_goal(self, index_lock, local_start, polygons, xmin, xmax, ymin, ymax,
                           target_altitude, min_distance):
        """
        Choose a random goal point that is at least 'min_distance' meters away from the local_start.
        """
        while True:
            x_goal = np.random.uniform(xmin, xmax)
            y_goal = np.random.uniform(ymin, ymax)
            z_goal = target_altitude  # Assuming constant altitude

            if np.linalg.norm([x_goal - local_start[0], y_goal - local_start[1]]) >= min_distance:
                point = (x_goal, y_goal, z_goal)
                if not collides(index_lock, self.rtree_index, point, polygons):
                    return point

    def plot_map(self, data, polygons, to_keep, G, path):
        """
        Plot the map with obstacles, graph, and path.
        """
        fig, ax = plt.subplots(figsize=(12, 12))

        # Collect obstacle vertices
        obstacle_vertices = []
        for poly, height in polygons:
            if height >= 5:  # Adjust this condition based on your requirements
                x, y = poly.exterior.xy
                obstacle_vertices.append(np.column_stack((x, y)))

        # Plot obstacles
        for vertices in obstacle_vertices:
            ax.plot(vertices[:, 0], vertices[:, 1], "gray")

        # Plot sampled points
        if to_keep:
            to_keep_array = np.array(to_keep)
            ax.scatter(to_keep_array[:, 0], to_keep_array[:, 1], c='red', s=2)

        # Plot graph edges
        edges = []
        for (n1, n2) in G.edges:
            edges.append([n1[:2], n2[:2]])  # Only use the first two coordinates (x, y)
        edge_collection = LineCollection(edges, colors='blue', linewidths=0.5, alpha=0.5)
        ax.add_collection(edge_collection)

        # Plot path
        if path is not None:
            path_points = np.array(path)
            ax.plot(path_points[:, 0], path_points[:, 1], 'green', linewidth=3)

        plt.xlabel('North', fontsize=20)
        plt.ylabel('East', fontsize=20)
        plt.title('Map with Obstacles, Graph, and Path', fontsize=20)
        plt.show()

    def plot_graph_optimised(self, G, start_point, goal_point, polygons):
        start_time = time.time()  # Record the start time for the entire function

        fig, ax = plt.subplots(figsize=(12, 12))

        if not G.nodes or not G.edges:
            print("Warning: Graph is empty or incomplete")
            return

        # Extract edges and convert to 2D
        edges = [((n1[0], n1[1]), (n2[0], n2[1])) for (n1, n2) in G.edges]

        # Create a LineCollection for the edges
        edge_collection = LineCollection(edges, colors='gray', linewidths=0.5)
        ax.add_collection(edge_collection)

        # Plot nodes in 2D
        nodes = np.array(G.nodes)
        ax.scatter(nodes[:, 0], nodes[:, 1], c='blue', s=10, zorder=2)

        # Plot the start and goal points
        ax.scatter(start_point[0], start_point[1], c='green', marker='o', s=100, label='Start', zorder=3)
        ax.scatter(goal_point[0], goal_point[1], c='red', marker='o', s=100, label='Goal', zorder=3)

        # Plot obstacles
        for polygon, height in polygons:
            if isinstance(polygon, shapely.geometry.MultiPolygon):
                for p in polygon:
                    ax.plot(*p.exterior.xy, 'black', zorder=1)
            elif isinstance(polygon, shapely.geometry.Polygon):
                ax.plot(*polygon.exterior.xy, 'black', zorder=1)

        plt.xlabel('East')
        plt.ylabel('North')
        plt.title('Graph with Start and Goal Points')
        plt.legend()
        plt.show()

        end_time = time.time()  # Record the end time for the entire function
        print(f"plot_graph executed in {end_time - start_time:.2f} seconds")  # duration for the entire function

        return fig, ax

    def add_path_to_plot(self, fig, ax, path):
        """
        Add the path to an existing plot.
        """
        path_points = np.array(path)
        ax.plot(path_points[:, 0], path_points[:, 1], 'green', linewidth=3)
        fig.canvas.draw()  # Ensure the canvas is updated

    def check_waypoints_validity(self, waypoints, data, polygons):
        xmin, xmax, ymin, ymax, zmin, zmax = self.get_bounds(data)

        for wp in waypoints:
            x, y, z, _ = wp
            print(f"Checking waypoint: {wp}")

            # Check within boundaries
            if not (xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax):
                print(f"Waypoint {wp} is out of bounds.")
                return False

            # Check for obstacles
            point = (x, y, z)
            for poly, height in polygons:
                if poly.contains(Point(x, y)) and height > z:
                    print(f"Waypoint {wp} collides with an obstacle.")
                    return False

        return True

    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Planning a path... ")
        TARGET_ALTITUDE = 5
        SAFETY_DISTANCE = 5

        self.target_position[2] = TARGET_ALTITUDE

        # Load colliders data
        filename = 'colliders.csv'
        with open(filename, 'r') as file:
            first_line = file.readline()

        # Split first line into lat and long strings
        lat0, lon0 = first_line.strip().split(',')

        # Get floating point values
        lat0 = np.float64(lat0.split()[1])
        lon0 = np.float64(lon0.split()[1])
        print(lat0, lon0)

        # set home position to (lon0, lat0, 0)
        self.set_home_position(lon0, lat0, 0)

        # retrieve current global position
        print(f"Global Position: {self.global_position}")

        # convert to current local position using global_to_local()
        local_start = global_to_local(self.global_position, self.global_home)
        print(f'global home {self.global_home}, position {self.global_position}, local position {self.local_position}')

        index_lock2 = threading.Lock()
        # adapt to set goal as latitude / longitude position and convert
        local_goal = self.choose_random_goal(index_lock2, local_start, polygons, xmin, xmax,
                                             ymin, ymax, TARGET_ALTITUDE, min_distance=100)

        start_point = closest_point(G, local_start)
        goal_point = closest_point(G, local_goal)
        print(f'Start Point: {start_point}, Goal Point: {goal_point}')

        # plot graph with start, goal and traversable edges.
        # fig, ax = self.plot_graph_optimised(G, start_point, goal_point, self.polygons)
        # print("Graph plotted")

        print("Searching for a path ...")
        try:
            path = a_star_graph(G, start_point, goal_point)
            if not path:
                print("No path found")
        except NetworkXNoPath as e:
            print(f"No path found: {e}")

        print("Path found: {0}".format(path))

        # path = combined_pruning(path)
        # print("Path pruned: ", path)

        # Add path to plot
        # self.add_path_to_plot(fig, ax, path)
        # print("Path added to plotted Graph")
        # plt.show()

        waypoints = [[int(local_start[0]), int(local_start[1]), int(TARGET_ALTITUDE), 0]]

        waypoints.extend([[int(p[0]), int(p[1]), int(TARGET_ALTITUDE), 0] for p in path])
        print("Waypoints:", waypoints)
        self.waypoints = waypoints

        self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #     pass

        self.stop_log()


if __name__ == "__main__":
    # Load colliders data
    filename = 'colliders.csv'
    data = np.loadtxt(filename, delimiter=',', dtype='Float64', skiprows=2)


    # Extract polygons before creating the connection
    def extract_polygons(data):
        polygons = []
        for i in range(data.shape[0]):
            # center north, east etc and half width north, east etc
            # of obstacles.
            north, east, alt, d_north, d_east, d_alt = data[i, :]

            # required calculations to calculate corner coordinates
            # from obstacle center.
            obstacle = [
                int(north - d_north),
                int(north + d_north),
                int(east - d_east),
                int(east + d_east)
            ]

            # corners will be:
            # bl = n - dn, e - de, br = n - dn, e + de
            # tr = n + dn, e + de, tl = n + dn, e - de
            corners = [(obstacle[0], obstacle[2]), (obstacle[0], obstacle[3]),
                       (obstacle[1], obstacle[3]), (obstacle[1], obstacle[2])]

            height = int(alt + d_alt)

            p = Polygon(corners)
            polygons.append((p, height))

        return polygons


    polygons = extract_polygons(data)
    print("Polygons extracted.")

    def merge_polygons(polygons):
        # Calculate the maximum height
        max_height = max(height for _, height in polygons)

        # Merge polygons using unary_union
        merged_polygon = unary_union([poly for poly, _ in polygons])

        return merged_polygon, max_height


    merged_polygon, merged_height = merge_polygons(polygons)
    print("Number of merged polygons:", len(merged_polygon))
    polygons = [(merged_polygon, merged_height)]
    print("Number of polygons:", len(polygons))

    # Create an R-tree index for polygons before the connection
    index = rtree.index.Index()
    for i, (poly, height) in enumerate(polygons):
        if isinstance(poly, shapely.geometry.MultiPolygon):
            for p in poly:
                index.insert(i, p.bounds)
        elif isinstance(poly, shapely.geometry.Polygon):
            index.insert(i, poly.bounds)
    print("R-tree constructed.")

    # set number of samples and generate them, finally put them into a list
    num_samples = 500


    def get_bounds(data):
        """
        Find bounds for the operational area based on obstacle data.
        """
        xmin = np.min(data[:, 0] - data[:, 3])
        xmax = np.max(data[:, 0] + data[:, 3])
        ymin = np.min(data[:, 1] - data[:, 4])
        ymax = np.max(data[:, 1] + data[:, 4])
        zmin = 0
        zmax = 100  # Example z bounds
        return xmin, xmax, ymin, ymax, zmin, zmax


    def collides(index_lock, polygons_index, point, polygons):
        point_obj = Point(point[0], point[1])
        with index_lock:  # Ensure thread-safe access to the R-tree index
            for idx in polygons_index.intersection((point[0], point[1], point[0], point[1])):
                poly, height = polygons[idx]
                if poly.contains(point_obj) and height > point[2]:
                    return True
        return False


    def can_connect(p1, p2, polygons):
        line = LineString([p1, p2])
        line_bbox = box(min(p1[0], p2[0]), min(p1[1], p2[1]), max(p1[0], p2[0]), max(p1[1], p2[1]))

        for poly, _ in polygons:
            poly_bbox = poly.bounds
            poly_bbox = box(poly_bbox[0], poly_bbox[1], poly_bbox[2], poly_bbox[3])

            if line_bbox.intersects(poly_bbox):
                if line.intersects(poly):
                    return False
        return True


    def create_graph_parallel(to_keep, radius, free_samples_tree, polygons):
        """
        Create a graph in parallel.
        """
        start_time = time.time()

        def process_point(index):
            point = to_keep[index]
            neighbors = free_samples_tree.query_ball_point(point, r=radius)
            edges = []
            for neighbor_index in neighbors:
                neighbor = to_keep[neighbor_index]
                if point != neighbor and can_connect(point, neighbor, polygons):
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


    xmin, xmax, ymin, ymax, zmin, zmax = get_bounds(data)

    xvals = np.random.uniform(xmin, xmax, num_samples)
    yvals = np.random.uniform(ymin, ymax, num_samples)
    zvals = np.random.uniform(zmin, zmax, num_samples)
    samples = list(zip(xvals, yvals, zvals))
    print("Samples Generated")

    # Parallelize collision checking
    index_lock = threading.Lock()
    with ThreadPoolExecutor() as executor:
        collision_results = list(executor.map(lambda p: collides(index_lock, index, p, polygons), samples))

    # Filter points that do not collide
    to_keep = [point for point, collides in zip(samples, collision_results) if not collides]
    print("To-keep Generated")

    # Convert the list to_keep to a NumPy array
    to_keep_array = np.array(to_keep)
    print("NP Array constructed")

    # Build the KDTree with obstacles in free space
    free_samples_tree = KDTree(to_keep_array)
    print("KDTree Created")

    # Define a radius for neighbor search
    radius = 300

    # create graph
    G = create_graph_parallel(to_keep, radius, free_samples_tree, polygons)
    print("Graph Created")

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=1200)
    drone = MotionPlanning(conn, precomputed_polygons=polygons, precomputed_rtree_index=index, colliders_data=data)

    time.sleep(1)
    drone.start()
