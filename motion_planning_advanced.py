import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from enum import auto

import msgpack
import shapely
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import KDTree
from shapely.geometry import Polygon, LineString, box
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.frame_utils import global_to_local
from udacidrone.messaging import MsgID

from planning_utils import *
from polygon import create_polygon
from nodes import create_nodes_and_graph


class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):

    def __init__(self,
                 connection,
                 precomputed_polygons=None,
                 precomputed_rtree_index=None,
                 colliders_data=None,
                 node_graph=None,
                 num_nodes=None,
                 radius=None,
                 conn_port=None,
                 conn_host=None,
                 persistent_local_goal=None):

        super().__init__(connection)

        self.num_nodes = num_nodes
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}
        self.polygons = precomputed_polygons
        self.rtree_index = precomputed_rtree_index
        self.data = colliders_data
        self.node_graph = node_graph
        self.radius = radius
        self.port = conn_port
        self.host = conn_host
        self.local_goal = persistent_local_goal
        self.waypoints_executed = False

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
        start_time = time.time()

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

        end_time = time.time()
        print(f"plot_graph executed in {end_time - start_time:.2f} seconds")

        return fig, ax

    def add_path_to_plot(self, fig, ax, path):
        """
        Add the path to an existing plot.
        """
        path_points = np.array(path)
        ax.plot(path_points[:, 0], path_points[:, 1], 'green', linewidth=3)
        fig.canvas.draw()  # Ensure the canvas is updated

    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Planning a path... ")
        TARGET_ALTITUDE = 5

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

        # Set home position to (lon0, lat0, 0)
        self.set_home_position(lon0, lat0, 0)

        # Retrieve current global position
        print(f"Global Position: {self.global_position}")

        # Convert to current local position using global_to_local()
        local_start = global_to_local(self.global_position, self.global_home)
        print(f'global home {self.global_home}, position {self.global_position}, local position {self.local_position}')

        # TODO: this code is duplicated in Nodes find a way to remove it here
        # Get boundaries of the mapped area (defined by the obstacle data)
        xmin, xmax, ymin, ymax, zmin, zmax = get_bounds(map_data)

        # Set local goal as latitude / longitude position and convert
        if self.local_goal is None:
            index_lock2 = threading.Lock()

            self.local_goal = self.choose_random_goal(index_lock2, local_start, polygons, xmin, xmax,
                                                 ymin, ymax, TARGET_ALTITUDE, min_distance=100)

        print(f"Local Goal: {self.local_goal}")

        start_point = closest_point(self.node_graph, local_start)
        goal_point = closest_point(self.node_graph, self.local_goal)
        print(f'Start Point: {start_point}, Goal Point: {goal_point}')

        # # plot graph with start, goal and traversable edges (THIS WILL CAUSE CONNECTION TO TIMEOUT)
        # fig, ax = self.plot_graph_optimised(self.node_graph, start_point, goal_point, self.polygons)
        # print("Graph plotted")

        path_found = False
        while not path_found:
            print("Searching for a path ...")
            try:
                path = a_star_graph(self.node_graph, start_point, goal_point)
                print("Path found")
                path_found = True
            except nx.NetworkXNoPath as e:
                print(f"No path found: {e}")
                self.stop()
                return

        pruned_path = combined_pruning(path)

        # # Add path to plot
        # self.add_path_to_plot(fig, ax, path)
        # print("Path added to plotted Graph")
        # plt.show()

        # waypoints = [[int(local_start[0]), int(local_start[1]), int(TARGET_ALTITUDE), 0]]

        waypoints = [[int(p[0]), int(p[1]), int(TARGET_ALTITUDE), 0] for p in pruned_path]
        print("Waypoints:", waypoints)
        self.waypoints = waypoints
        self.waypoints_executed = True
        self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #     pass

        self.stop_log()
        return self.local_goal, self.waypoints_executed


if __name__ == "__main__":
    print("Starting Preprocessing Tasks")

    # Set number of node points to create and consider for path planning.
    num_nodes = 500

    # Define maximum radius for neighbor search
    radius = 300

    # Load colliders data
    filename = 'colliders.csv'
    map_data = np.loadtxt(filename, delimiter=',', dtype='Float64', skiprows=2)

    # Set initial local_goal point as None for first loop, successive loops will use local_goal
    persistent_local_goal = None

    # Loop until waypoint is executed (Polygon, Node and Graph creation have to occur outside
    # of API connection to avoid timeout)
    while True:
        # Create polygons and rtree
        polygons, rtree = create_polygon(map_data)

        # Crate world graph and nodes to consider for path planning
        graph = create_nodes_and_graph(map_data, num_nodes, rtree, polygons, radius)

        # Create Connection
        port = 5760
        host = "127.0.0.1"
        conn = MavlinkConnection(f"tcp:{host}:{port}", timeout=1200)

        drone = MotionPlanning(conn,
                               precomputed_polygons=polygons,
                               precomputed_rtree_index=rtree,
                               colliders_data=map_data,
                               node_graph=graph,
                               num_nodes=num_nodes,
                               radius=radius,
                               conn_host=host,
                               conn_port=port,
                               persistent_local_goal=persistent_local_goal)

        persistent_local_goal, path_executed = drone.start()

        if path_executed:
            break

