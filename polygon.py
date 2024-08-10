import rtree
import shapely
from shapely.ops import unary_union
from shapely.geometry import Polygon


def merge_polygons(polygons_list):
    """
    # Merges polygons into one to speed up computation time
    """
    # Calculate the maximum height
    max_height = max(height for _, height in polygons_list)

    # Merge polygons using unary_union
    polygon_merged = unary_union([poly for poly, _ in polygons_list])

    # print("Number of merged polygons:", len(polygon_merged))
    return polygon_merged, max_height


def extract_polygons(data, safety_margin=5):
    """
    # Extract building polygon representations from collision data
    """
    building_polygons = []
    for i in range(data.shape[0]):
        # center north, east etc. and half width north, east etc. of obstacles.
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

        p = Polygon(corners).buffer((safety_margin))
        building_polygons.append((p, height))

    # print("Polygons extracted.")
    return building_polygons


def create_rtree_index(polygons):
    index = rtree.index.Index()
    for i, (poly, height) in enumerate(polygons):
        if isinstance(poly, shapely.geometry.MultiPolygon):
            for p in poly:
                index.insert(i, p.bounds)
        elif isinstance(poly, shapely.geometry.Polygon):
            index.insert(i, poly.bounds)
    # print("R-tree constructed.")
    return index


def create_polygon(collision_data):
    polygons = extract_polygons(collision_data, safety_margin=5)
    merged_polygon, merged_height = merge_polygons(polygons)
    polygons = [(merged_polygon, merged_height)]
    rtree = create_rtree_index(polygons)
    return polygons, rtree
