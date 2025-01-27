import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
from scipy.spatial import KDTree
from collections import defaultdict
import gdsfactory as gf
import uuid
import sys


class Edge:
    def __init__(self, point1, point2, polygon_id=0):
        self.point1 = point1
        self.point2 = point2
        self.polygon_id = polygon_id
        self.midpoint = tuple((np.array(point1) + np.array(point2)) / 2)
    
    def __eq__(self, other):
        """Check if two edges are the same, accounting for order."""
        return (
            (np.allclose(self.point1, other.point1) and np.allclose(self.point2, other.point2)) or
            (np.allclose(self.point1, other.point2) and np.allclose(self.point2, other.point1))
        )

    def __hash__(self):
        """Hash for using Edge in sets or dictionaries."""
        return hash((tuple(sorted((tuple(self.point1), tuple(self.point2))))))

class Polygon:
    
    def __init__(self, edges_or_vertices, neighbors=None):
        if isinstance(edges_or_vertices, list) and all(isinstance(e, Edge) for e in edges_or_vertices):
            # Initialize from a list of Edge objects
            self.edges = edges_or_vertices
        elif isinstance(edges_or_vertices, np.ndarray):
            # Initialize from a numpy array of vertices
            self.edges = []
            num_points = edges_or_vertices.shape[0]
            for i, point in enumerate(edges_or_vertices):
                point1 = tuple(edges_or_vertices[i])
                point2 = tuple(edges_or_vertices[(i + 1) % num_points])
                edge = Edge(point1, point2)
                self.edges.append(edge)
        
        self.neighbors = neighbors if neighbors is not None else []
        self.merged = False
        self.visited = False
        pass
           
    def __eq__(self, other):
        """Check if two edges are the same, accounting for order."""
        return (
            set(self.edges) == set(other.edges)
        )
    
    def __hash__(self):
        h = 0
        for e in self.edges:
            h += hash(e)
        return h

    def merge(self, polygon):
        shared_edges = list(set(self.edges) & set(polygon.edges))
        combined_edges = list(set(self.edges + polygon.edges) - set(shared_edges))
        combined_neighbors = list(set(self.neighbors + polygon.neighbors) - set([self, polygon]))
        # edges = self.edges.copy()
        # edges.extend(polygon.edges)
        # edges = list(set(edges))        
        # neighbors = self.neighbors.copy()
        # neighbors.extend(polygon.neighbors)
        # neighbors = list(set(neighbors))
        for n in combined_neighbors[:]:
            if n.merged:
                combined_neighbors.remove(n)
        merged_polygon = Polygon(combined_edges, combined_neighbors)
        # neighbors = []
        # for n in merged_polygon.neighbors:
        #     if n != self and n != polygon:
        #         neighbors.append(n)
        # merged_polygon.neighbors = neighbors

        self.merged = True
        polygon.merged = True
        return merged_polygon
    
    def copy(self):
        edges = self.edges.copy()                
        neighbors = self.neighbors.copy()
                       
        return Polygon(edges, neighbors)

    def get_vertices(self):
        vertices = []
        for edge in self.edges:
            vertices.append(edge.point1)
            vertices.append(edge.point2)
        vertices = list(set(vertices))
        vertices = np.array(vertices)
        for i, v in enumerate(vertices):
            vertices[i, :] = v
            # Calculate the centroid of the points
            centroid = np.mean(vertices, axis=0)
            angles = np.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])
            sorted_indices = np.argsort(angles)
            sorted_vertices = vertices[sorted_indices]
            # Step 4: Close the polygon by appending the first point to the end
            sorted_vertices = np.vstack([sorted_vertices, sorted_vertices[0]])
        return sorted_vertices

def polygon_neighbor_collect(polygon, connected_polygon_list):
    if not polygon.visited:
        polygon.visited = True
        connected_polygon_list.append(polygon)
        for n in polygon.neighbors:
            polygon_neighbor_collect(n, connected_polygon_list)
        return
    return
    
def get_polygon_family(polygon_list):
    polygon_family_list = []
    total_polygons = len(polygon_list)
    # Step 1: Collect neighbors into families
    for idx, p in enumerate(polygon_list):
        if not p.visited:
            connected_polygon_list = []
            polygon_neighbor_collect(p, connected_polygon_list)
            if len(connected_polygon_list) > 0:
                polygon_family_list.append(connected_polygon_list)
            else:
                polygon_family_list.append([p])
        
        # Print status update for collecting neighbors
        if idx % 1000 == 0:
            sys.stdout.write(f'\rCollecting polygon families: {idx + 1}/{total_polygons} completed')
            sys.stdout.flush()
        # time.sleep(0.01)  # This sleep is optional, just for demonstration purposes
        
    print(f"\nFinished collecting polygon families. Total polygon families found: {len(polygon_family_list)}")
    return polygon_family_list
    

def combine_polygons(polygon_family_list):
    combined_polygon_list = []
    total_families = len(polygon_family_list)    
    # Step 2: Combine the polygons in each family
    for idx, f in enumerate(polygon_family_list):
        if len(f) == 0:
            continue        
        # p_combined = f.pop(0)
        if len(f) == 1:
            combined_polygon_list.append(f[0])
            continue
        edges = []
        for p in f:
            edges.extend(p.edges)
        edges = list(set(edges))
        p_combined = Polygon(edges)
            # Merge each polygon in the family
            # p_combined = p_combined.merge(p)
        combined_polygon_list.append(p_combined)        
        # Print status update for combining families
        if idx % 1000 == 0:
            sys.stdout.write(f'\rCombining polygon families: {idx + 1}/{total_families} completed')
            sys.stdout.flush()        

    print("\nFinished combining all polygon families.")
    
    return combined_polygon_list

def extract_edges_from_polygons(polygons):
    start_time = time.time()
    edges = []
    all_points = []
    edge_map = defaultdict(list)  # Map from edges to their polygon IDs
    polygon_list = []
    # Step 1: Extract edges directly from ordered points
    print("Starting edge extraction from polygons...")
    for poly_id, points in enumerate(polygons):
        if (poly_id + 1) % 1000 == 0 or poly_id == len(polygons) - 1:
            print(f"\rExtracting edges from polygon {poly_id + 1}/{len(polygons)}...", end="", flush=True)
        num_points = len(points)
        polygon_edges = []
        for i in range(num_points):
            point1 = tuple(points[i])
            point2 = tuple(points[(i + 1) % num_points])
            edge = Edge(point1, point2, poly_id)
            edges.append(edge)
            polygon_edges.append(edge)
            all_points.append(edge.midpoint)
            edge_map[edge].append(poly_id)
        polygon_list.append(Polygon(polygon_edges))
    print(f"\nEdge extraction completed. Total edges extracted: {len(edges)}")
    print(f"Time taken for edge extraction: {time.time() - start_time:.2f} seconds")

    return edges, all_points, edge_map, polygon_list

def connect_polygons(edges, all_points, polygon_list):
    start_time = time.time()
    # Step 2: Build KD-Tree with edge midpoints
    print("Building KD-Tree with edge midpoints...")
    midpoints = np.array(all_points)
    kd_tree = KDTree(midpoints)
    print("KD-Tree built successfully.")
    # Step 3: Detect shared edges and merge polygons
    print("Detecting shared edges and merging polygons...")
    visited = set()  # To avoid processing the same edge twice    
    connected_polygons = 0
    shared_edges = []
    for edge in edges:
        if edge in visited:
            continue
        # Query KD-Tree for the nearest edge
        dist, idx = kd_tree.query(edge.midpoint)
        nearest_edge = edges[idx]
        # Check if they are the same edge
        if edge != nearest_edge or edge.polygon_id == nearest_edge.polygon_id:
            continue
        shared_edges.append(edge)
        polygon_list[edge.polygon_id].neighbors.append(polygon_list[nearest_edge.polygon_id])
        polygon_list[nearest_edge.polygon_id].neighbors.append(polygon_list[edge.polygon_id])
        connected_polygons += 1
        # Mark both edges as visited
        visited.add(edge)
        visited.add(nearest_edge)
    print(f"Polygon connection discovery completed. Total connected polygons found: {connected_polygons}")
    print(f"Time taken for polygon combination: {time.time() - start_time:.2f} seconds")
    return polygon_list, shared_edges

def plot_generated_polygons(polygons):
    """Plot the generated polygons."""
    plt.figure(figsize=(8, 6))
    color_map = plt.colormaps['tab10']  # Using 'tab10' colormap to have different colors for each polygon

    for idx, p in enumerate(polygons):
        color = color_map(idx % 10)  # Get a color from the colormap
        for edge_idx, e in enumerate(p.edges):
            edge_vertices = np.vstack([e.point1, e.point2])
            # Only add the label for the first edge of each polygon to avoid duplicate labels
            if edge_idx == 0:
                plt.plot(
                    edge_vertices[:, 0],
                    edge_vertices[:, 1],
                    marker='o',
                    color=color,
                    label=f'Polygon {idx}'
                )
            else:
                plt.plot(
                    edge_vertices[:, 0],
                    edge_vertices[:, 1],
                    marker='o',
                    color=color
                )

    plt.legend()
    plt.title("Generated Test Polygons")
    plt.axis('equal')
    plt.show()

def plot_generated_polygons_vertices(polygons_vertices):
    """Plot the generated polygons."""
    plt.figure(figsize=(8, 6))
    for idx, polygon in enumerate(polygons_vertices):
        plt.plot(
            np.append(polygon[:, 0], polygon[0, 0]),
            np.append(polygon[:, 1], polygon[0, 1]),
            marker='o',
            label=f'Polygon {idx}'
        )
    plt.legend()
    plt.title("Generated Test Polygons")
    plt.axis('equal')
    plt.show()


def create_gds_from_polygons(polygons, gds_filename="combined_polygons.gds"):
    """Create a GDS file from combined polygons using gdsfactory."""
    # Create a blank component with a unique name to avoid duplication
    component = gf.Component(f"combined_polygons_{uuid.uuid4().hex}")

    # Add each polygon to the component
    for idx, polygon in enumerate(polygons):
        # Create a GDS-compatible polygon by converting numpy array to list of tuples
        polygon_points = [tuple(map(float, point)) for point in polygon]
        # Add the polygon to the component
        component.add_polygon(polygon_points, layer=(1, 0))  # Layer 1, datatype 0

    # Write the component to a GDS file
    component.write_gds(gds_filename)
    print(f"GDS file '{gds_filename}' created successfully with {len(polygons)} polygons.")

def visit_neighbors(polygon):
    if not polygon.visited:
        polygon.visited = True
        for n in polygon.neighbors:
            visit_neighbors(n)
        return
    return
    
def check_visited(polygon_list):
    visited_num = 0
    for p in polygon_list:
        if p.visited:
            visited_num += 1
    print(f'{visited_num} / {len(polygon_list)} visited')
