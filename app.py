from flask import Flask, render_template_string, request, jsonify
import osmnx as ox
import traceback
from pyproj import Transformer
import heapq
from collections import deque


ox.settings.log_console = True
ox.settings.use_cache = True

app = Flask(__name__, static_folder='static')


# ============== GRAPH ALGORITHMS (Implemented from scratch) ==============

def get_neighbors(graph, node):
    """Get all neighbors of a node in the graph."""
    return list(graph.successors(node))


def get_edge_weight(graph, u, v):
    """Get the weight (length) of edge between u and v."""
    try:
        data = graph.get_edge_data(u, v)
        if data:
            first_edge = list(data.values())[0]
            return first_edge.get('length', 1)
        return float('inf')
    except:
        return float('inf')


def heuristic(node, goal, nodes_gdf):
    """
    A* heuristic: Euclidean distance between node and goal.
    Uses latitude/longitude from nodes GeoDataFrame.
    """
    try:
        lat1 = nodes_gdf.loc[node].y
        lng1 = nodes_gdf.loc[node].x
        lat2 = nodes_gdf.loc[goal].y
        lng2 = nodes_gdf.loc[goal].x
        
        # Simple Euclidean distance (approximation for small distances)
        return ((lat2 - lat1) ** 2 + (lng2 - lng1) ** 2) ** 0.5
    except:
        return 0


def astar(graph, start, goal, nodes_gdf):
    """
    A* Algorithm Implementation from scratch.
    Returns the shortest path as a list of nodes.
    """
    # Priority queue: (f_score, node)
    open_set = [(0, start)]
    
    came_from = {}
    
    g_score = {start: 0}
    
    f_score = {start: heuristic(start, goal, nodes_gdf)}
    
    visited = set()
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        
        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        
        if current in visited:
            continue

        visited.add(current)

        # Check neighbors
        for neighbor in get_neighbors(graph, current):
            if neighbor in visited:
                continue
            
            tentative_g = g_score[current] + get_edge_weight(graph, current, neighbor)
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal, nodes_gdf)
                f_score[neighbor] = f
                heapq.heappush(open_set, (f, neighbor))
    
    # No path found
    raise Exception("A* could not find a path")











def dijkstra(graph, start, goal):
    """
    Dijkstra's Algorithm Implementation from scratch.
    Returns the shortest path as a list of nodes.
    """
    # Priority queue: (distance, node)
    pq = [(0, start)]
    
    # Track distances
    distances = {start: 0}
    
    # Track where we came from
    came_from = {}
    
    # Set to track visited nodes
    visited = set()
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        
        if current in visited:
            continue
        
        visited.add(current)
        
        # Check neighbors
        for neighbor in get_neighbors(graph, current):
            if neighbor in visited:
                continue
            
            new_dist = current_dist + get_edge_weight(graph, current, neighbor)
            
            if neighbor not in distances or new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                came_from[neighbor] = current
                heapq.heappush(pq, (new_dist, neighbor))
    
    # No path found
    raise Exception("Dijkstra could not find a path")


def bfs(graph, start, goal):
    """
    Breadth-First Search Implementation from scratch.
    Returns any path (not necessarily shortest) as a list of nodes.
    """
    queue = deque([(start, [start])])
    visited = set([start])

    while queue:
        current, path = queue.popleft()

        if current == goal:
            return path

        for neighbor in get_neighbors(graph, current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    # No path found
    raise Exception("BFS could not find a path")


def ucs(graph, start, goal):
    """
    Uniform Cost Search Implementation from scratch.
    Returns the shortest path as a list of nodes.
    Similar to Dijkstra but stops when goal is reached.
    """
    # Priority queue: (cost, node)
    pq = [(0, start)]

    # Track costs
    costs = {start: 0}

    # Track where we came from
    came_from = {}

    # Set to track visited nodes
    visited = set()

    while pq:
        current_cost, current = heapq.heappop(pq)

        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        if current in visited:
            continue

        visited.add(current)

        # Check neighbors
        for neighbor in get_neighbors(graph, current):
            if neighbor in visited:
                continue

            new_cost = current_cost + get_edge_weight(graph, current, neighbor)

            if neighbor not in costs or new_cost < costs[neighbor]:
                costs[neighbor] = new_cost
                came_from[neighbor] = current
                heapq.heappush(pq, (new_cost, neighbor))

    # No path found
    raise Exception("UCS could not find a path")


def has_path(graph, start, goal):
    """
    Check if a path exists between start and goal using BFS.
    """
    if start not in graph or goal not in graph:
        return False
    
    visited = set([start])
    queue = deque([start])
    
    while queue:
        current = queue.popleft()
        
        if current == goal:
            return True
        
        for neighbor in get_neighbors(graph, current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return False


def weakly_connected_components(graph):
    """
    Find all weakly connected components in a directed graph.
    Treats the graph as undirected for connectivity.
    """
    visited = set()
    components = []
    
    def bfs_component(start):
        """BFS to find all nodes in the same component."""
        component = set()
        queue = deque([start])
        component.add(start)
        
        while queue:
            node = queue.popleft()
            
            # Check both successors and predecessors (weakly connected)
            for neighbor in list(graph.successors(node)) + list(graph.predecessors(node)):
                if neighbor not in component:
                    component.add(neighbor)
                    queue.append(neighbor)
        
        return component
    
    for node in graph.nodes():
        if node not in visited:
            component = bfs_component(node)
            visited.update(component)
            components.append(component)
    
    return components


def single_source_dijkstra_path_length(graph, source):
    """
    Dijkstra's algorithm to find shortest distances from source to all nodes.
    Returns a dictionary {node: distance}.
    """
    distances = {source: 0}
    pq = [(0, source)]
    visited = set()
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if current in visited:
            continue
        
        visited.add(current)
        
        for neighbor in get_neighbors(graph, current):
            if neighbor in visited:
                continue
            
            new_dist = current_dist + get_edge_weight(graph, current, neighbor)
            
            if neighbor not in distances or new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))
    
    return distances


# ============== PATHFINDER CLASS ==============

class PathFinder:
    def __init__(self):
        self.graph = None
        self.graph_proj = None
        self.nodes = None
        self.edges = None
        self.region_name = None
        self.transformer = None

    def load_region(self, region_name):
        """
        Download the drive network for the specified Algerian region,
        extract the largest weakly connected component,
        project the graph, compute its centroid, and then truncate the
        unprojected graph to a 10 km radius using network distances.
        """
        try:
            self.region_name = region_name
            print(f"Loading region: {region_name}, Algeria...")
            # Download network for the region
            self.graph = ox.graph_from_place(f"{region_name}, Algeria", network_type="drive")

            # Keep the largest weakly connected component (using our implementation)
            components = weakly_connected_components(self.graph)
            largest_cc = max(components, key=len)
            self.graph = self.graph.subgraph(largest_cc).copy()

            # Project the graph to a suitable CRS (UTM)
            self.graph_proj = ox.project_graph(self.graph)

            # Create a transformer to convert coordinates from EPSG:4326 to the graph's CRS
            self.transformer = Transformer.from_crs("EPSG:4326", self.graph_proj.graph["crs"], always_xy=True)

            # Get GeoDataFrames from the unprojected graph (for display)
            self.nodes, self.edges = ox.graph_to_gdfs(self.graph)

            # Compute centroid from the projected nodes
            nodes_proj, _ = ox.graph_to_gdfs(self.graph_proj)
            center_x = nodes_proj['x'].mean()
            center_y = nodes_proj['y'].mean()
            center_node = ox.nearest_nodes(self.graph_proj, center_x, center_y)

            # Truncate the graph: use Dijkstra to get network distances from the center node
            max_distance = 10000  # 10 km in meters
            lengths = single_source_dijkstra_path_length(self.graph, center_node)
            nodes_within = [node for node, d in lengths.items() if d <= max_distance]
            self.graph = self.graph.subgraph(nodes_within).copy()

            # Update GeoDataFrames after truncation
            self.nodes, self.edges = ox.graph_to_gdfs(self.graph)
            print(f"Graph loaded and truncated: {len(self.nodes)} nodes, {len(self.edges)} edges")
            return True
        except Exception as e:
            print(f"Error loading region: {e}")
            traceback.print_exc()
            return False

    def get_region_bounds(self):
        """Return the bounding box of the loaded graph with a small buffer."""
        if self.graph is None:
            return None
        try:
            north = float(self.nodes.y.max())
            south = float(self.nodes.y.min())
            east = float(self.nodes.x.max())
            west = float(self.nodes.x.min())
            buffer = 0.01  # approximate buffer (in degrees)
            return {
                'north': north + buffer,
                'south': south - buffer,
                'east': east + buffer,
                'west': west - buffer
            }
        except Exception as e:
            print(f"Error getting region bounds: {e}")
            traceback.print_exc()
            return None

    def find_nearest_node(self, lat, lng):
        """
        Convert the (lat, lng) in EPSG:4326 to the projected CRS and
        use the projected graph to find the nearest node.
        """
        if self.graph is None or self.graph_proj is None or self.transformer is None:
            return None
        try:
            # Convert (lng, lat) to projected coordinates (note the order: lng, lat)
            proj_x, proj_y = self.transformer.transform(lng, lat)
            nearest_node = ox.nearest_nodes(self.graph_proj, proj_x, proj_y)
            print(f"Nearest node to ({lat}, {lng}) is {nearest_node}")
            return nearest_node
        except Exception as e:
            print(f"Error in find_nearest_node: {e}")
            traceback.print_exc()
            return None

    def find_shortest_path(self, start_node, end_node, algorithm='astar'):
        """
        Compute the shortest path between two nodes using the specified algorithm.
        Returns a list of [lat, lng] coordinate pairs (from the unprojected graph)
        and the total path length in meters.
        """
        if self.graph is None:
            return None, 0
        try:
            if start_node not in self.graph or end_node not in self.graph:
                print("Start or end node not in graph.")
                return None, 0
            if not has_path(self.graph, start_node, end_node):
                print("No path exists between the selected nodes.")
                return None, 0

            # Use the selected algorithm
            print(f"Using {algorithm.upper()} algorithm...")
            if algorithm == 'astar':
                path = astar(self.graph, start_node, end_node, self.nodes)
            elif algorithm == 'dijkstra':
                path = dijkstra(self.graph, start_node, end_node)
            elif algorithm == 'bfs':
                path = bfs(self.graph, start_node, end_node)
            elif algorithm == 'ucs':
                path = ucs(self.graph, start_node, end_node)
            else:
                print(f"Unknown algorithm: {algorithm}, falling back to A*")
                path = astar(self.graph, start_node, end_node, self.nodes)

            # Calculate total path length
            path_length = 0
            for u, v in zip(path[:-1], path[1:]):
                data = self.graph.get_edge_data(u, v)
                if data:
                    first_edge = list(data.values())[0]
                    path_length += first_edge.get('length', 0)

            # Get path coordinates (unprojected lat/lng) for the frontend
            path_coords = []
            for node in path:
                if node in self.nodes.index:
                    lat_val = float(self.nodes.loc[node].y)
                    lng_val = float(self.nodes.loc[node].x)
                    path_coords.append([lat_val, lng_val])
            print(f"Path found with {len(path)} nodes, length: {path_length} meters")
            return path_coords, path_length
        except Exception as e:
            print(f"Error finding shortest path: {e}")
            traceback.print_exc()
            return None, 0

    def get_all_nodes(self):
        """Return all loaded nodes as a list of [lat, lng] pairs."""
        if self.nodes is None:
            return []
        return self.nodes.apply(lambda row: [row['y'], row['x']], axis=1).tolist()

    def get_available_regions(self):
        """Return a list of example Algerian regions."""
        return ['Algiers','Setif','Bejaia', 'Amizour','Oran', 'Constantine', 'Annaba', 'Tlemcen',
                 'Batna',  'Biskra', 'Blida',]

# Create a global PathFinder instance
path_finder = PathFinder()

# HTML Template with embedded JavaScript
BASE_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Algeria Path Finder</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
  <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
  <style>
    /* Optional styling for node markers */
    .node-marker {
      background-color: black;
      border-radius: 50%;
      width: 4px;
      height: 4px;
    }
    .button-group { margin-top: 10px; }

    /* Algorithm Selector Styling */
    .algorithm-selector {
      display: flex;
      align-items: center;
      gap: 15px;
      margin: 15px 0;
      padding: 15px;
      background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
      border-radius: 12px;
      flex-wrap: wrap;
    }

    .algorithm-selector > label {
      font-weight: 600;
      color: #4a5568;
      font-size: 1rem;
      min-width: 140px;
    }

    .algorithm-options {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }

    .algorithm-option {
      position: relative;
    }

    .algorithm-option input[type="checkbox"] {
      position: absolute;
      opacity: 0;
      cursor: pointer;
    }

    .algorithm-option label {
      display: inline-block;
      padding: 10px 20px;
      background: white;
      border: 3px solid #e2e8f0;
      border-radius: 8px;
      cursor: pointer;
      font-weight: 500;
      font-size: 14px;
      color: #4a5568;
      transition: all 0.3s ease;
      box-shadow: 0 2px 4px rgba(0,0,0,0.05);
      min-width: 80px;
      text-align: center;
      margin: 0;
    }

    /* A* Star - Purple */
    .algorithm-option:nth-child(1) input[type="checkbox"]:checked + label {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      border-color: #667eea;
      box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
      transform: translateY(-2px);
    }

    /* Dijkstra - Blue */
    .algorithm-option:nth-child(2) input[type="checkbox"]:checked + label {
      background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
      color: white;
      border-color: #4facfe;
      box-shadow: 0 4px 12px rgba(79, 172, 254, 0.4);
      transform: translateY(-2px);
    }

    /* UCS - Green */
    .algorithm-option:nth-child(3) input[type="checkbox"]:checked + label {
      background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
      color: white;
      border-color: #43e97b;
      box-shadow: 0 4px 12px rgba(67, 233, 123, 0.4);
      transform: translateY(-2px);
    }

    /* BFS - Orange */
    .algorithm-option:nth-child(4) input[type="checkbox"]:checked + label {
      background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
      color: white;
      border-color: #fa709a;
      box-shadow: 0 4px 12px rgba(250, 112, 154, 0.4);
      transform: translateY(-2px);
    }

    .algorithm-option label:hover {
      border-color: #667eea;
      box-shadow: 0 4px 8px rgba(102, 126, 234, 0.2);
    }

    .algorithm-info {
      font-size: 12px;
      color: #718096;
      margin-top: 8px;
      font-style: italic;
    }

    /* Algorithm Legend */
    .algorithm-legend {
      display: flex;
      gap: 15px;
      flex-wrap: wrap;
      margin-top: 10px;
      padding: 10px 15px;
      background: rgba(255, 255, 255, 0.9);
      border-radius: 8px;
      font-size: 13px;
    }

    .legend-item {
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .legend-color {
      width: 20px;
      height: 4px;
      border-radius: 2px;
    }

    .legend-color.astar { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .legend-color.dijkstra { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
    .legend-color.ucs { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }
    .legend-color.bfs { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); }
  </style>
</head>
<body>
  <div class="app-container">
    <header>
      <h1>Shortest Path Finder - Algeria</h1>
    </header>
    <div class="controls">
      <div class="form-group">
        <label for="region-select">Select Region (max distance loaded is 10km2):</label>
        <select id="region-select">
          {% for region in regions %}
            <option value="{{ region }}">{{ region }}</option>
          {% endfor %}
        </select>
        <button id="load-region-btn">Load Region</button>
      </div>
      
      <div class="algorithm-selector">
        <label>Algorithms:</label>
        <div class="algorithm-options">
          <div class="algorithm-option">
            <input type="checkbox" id="algo-astar" name="algorithm" value="astar" checked>
            <label for="algo-astar">A* Star</label>
          </div>
          <div class="algorithm-option">
            <input type="checkbox" id="algo-dijkstra" name="algorithm" value="dijkstra">
            <label for="algo-dijkstra">Dijkstra</label>
          </div>
          <div class="algorithm-option">
            <input type="checkbox" id="algo-ucs" name="algorithm" value="ucs">
            <label for="algo-ucs">UCS</label>
          </div>
          <div class="algorithm-option">
            <input type="checkbox" id="algo-bfs" name="algorithm" value="bfs">
            <label for="algo-bfs">BFS</label>
          </div>
        </div>
      </div>
      <p class="algorithm-info">Select one or more algorithms to compare their paths</p>
      <div class="algorithm-legend" id="algorithm-legend" style="display: none;">
        <div class="legend-item"><div class="legend-color astar"></div><span>A* Star</span></div>
        <div class="legend-item"><div class="legend-color dijkstra"></div><span>Dijkstra</span></div>
        <div class="legend-item"><div class="legend-color ucs"></div><span>UCS</span></div>
        <div class="legend-item"><div class="legend-color bfs"></div><span>BFS</span></div>
      </div>
      
      <div class="form-group">
        <button id="find-path-btn" disabled>Find Shortest Path</button>
        <button id="reset-btn" disabled>Reset Points</button>
      </div>
      <div class="button-group">
        <button id="toggle-nodes-btn" disabled>Hide Nodes</button>
      </div>
      <div id="status" class="status">Select a region and click 'Load Region'</div>
      <div id="path-info" class="path-info"></div>
    </div>
    <div id="map-container">
      <div id="map"></div>
    </div>
    <div class="instructions">
      <h3>Instructions:</h3>
      <ol>
        <li>Select an Algerian region from the dropdown and click "Load Region"</li>
        <li>Click on the map to choose a start point (green marker)</li>
        <li>Click again to choose an end point (blue marker)</li>
        <li>Click "Find Shortest Path" to calculate and display the route</li>
        <li>Click "Reset Points" to clear markers and try again</li>
        <li>Use "Hide Nodes"/"Show Nodes" to toggle display of all nodes</li>
      </ol>
      <p class="note">Note: For best results, select points on or near roads.</p>
    </div>
  </div>

  <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
  <script>
    document.addEventListener('DOMContentLoaded', function() {
      let map = null;
      let startMarker = null;
      let endMarker = null;
      let pathLayers = [];
      let nodesLayer = null;
      let isRegionLoaded = false;
      let areNodesVisible = true;

      const regionSelect = document.getElementById('region-select');
      const loadRegionBtn = document.getElementById('load-region-btn');
      const findPathBtn = document.getElementById('find-path-btn');
      const resetBtn = document.getElementById('reset-btn');
      const toggleNodesBtn = document.getElementById('toggle-nodes-btn');
      const statusDiv = document.getElementById('status');
      const pathInfoDiv = document.getElementById('path-info');

      function initMap(center = [36.7538, 3.0588], zoom = 12) {
        if (map) { map.remove(); }
        map = L.map('map').setView(center, zoom);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
          attribution: '&copy; OpenStreetMap contributors'
        }).addTo(map);
        map.on('click', onMapClick);
      }

      initMap();

      loadRegionBtn.addEventListener('click', function() {
        const region = regionSelect.value;
        statusDiv.textContent = `Loading ${region}...`;
        loadRegionBtn.disabled = true;
        resetPoints();
        const formData = new FormData();
        formData.append('region', region);
        fetch('/load_region', {
          method: 'POST',
          body: formData
        })
        .then(response => response.json())
        .then(data => {
          loadRegionBtn.disabled = false;
          if (data.success) {
            statusDiv.textContent = data.message + ' Click on the map to select points.';
            if (data.bounds) {
              map.fitBounds([
                [data.bounds.south, data.bounds.west],
                [data.bounds.north, data.bounds.east]
              ]);
            }
            // Add all nodes as small circle markers
            if (data.nodes && data.nodes.length > 0) {
              if (nodesLayer) { map.removeLayer(nodesLayer); }
              nodesLayer = L.layerGroup();
              data.nodes.forEach(coord => {
                L.circleMarker(coord, { radius: 2, color: 'black', fillOpacity: 1 })
                  .addTo(nodesLayer);
              });
              nodesLayer.addTo(map);
              areNodesVisible = true;
              toggleNodesBtn.textContent = 'Hide Nodes';
              toggleNodesBtn.disabled = false;
            }
            isRegionLoaded = true;
            resetBtn.disabled = false;
          } else {
            statusDiv.textContent = data.message;
            isRegionLoaded = false;
          }
        })
        .catch(error => {
          loadRegionBtn.disabled = false;
          console.error('Error:', error);
          statusDiv.textContent = 'Error loading region.';
        });
      });

      toggleNodesBtn.addEventListener('click', function() {
        if (!nodesLayer) return;
        if (areNodesVisible) {
          map.removeLayer(nodesLayer);
          toggleNodesBtn.textContent = 'Show Nodes';
          areNodesVisible = false;
        } else {
          nodesLayer.addTo(map);
          toggleNodesBtn.textContent = 'Hide Nodes';
          areNodesVisible = true;
        }
      });

      function onMapClick(e) {
        if (!isRegionLoaded) {
          statusDiv.textContent = 'Please load a region first.';
          return;
        }
        const latlng = e.latlng;
        if (!startMarker) {
          startMarker = L.marker(latlng, {
            icon: L.icon({
              iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-green.png',
              shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
              iconSize: [25, 41],
              iconAnchor: [12, 41]
            })
          }).addTo(map);
          startMarker.bindPopup('Start Point').openPopup();
          statusDiv.textContent = 'Start point selected. Now choose the end point.';
        } else if (!endMarker) {
          endMarker = L.marker(latlng, {
            icon: L.icon({
              iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-blue.png',
              shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
              iconSize: [25, 41],
              iconAnchor: [12, 41]
            })
          }).addTo(map);
          endMarker.bindPopup('End Point').openPopup();
          statusDiv.textContent = 'End point selected. Click "Find Shortest Path".';
          findPathBtn.disabled = false;
        }
      }

      findPathBtn.addEventListener('click', function() {
        if (!startMarker || !endMarker) {
          statusDiv.textContent = 'Select both start and end points.';
          return;
        }

        // Get selected algorithms
        const selectedAlgorithms = Array.from(document.querySelectorAll('input[name="algorithm"]:checked'))
          .map(cb => cb.value);

        if (selectedAlgorithms.length === 0) {
          statusDiv.textContent = 'Please select at least one algorithm.';
          return;
        }

        statusDiv.textContent = `Calculating paths with ${selectedAlgorithms.length} algorithm(s)...`;
        findPathBtn.disabled = true;

        // Clear previous paths
        if (pathLayers) {
          pathLayers.forEach(layer => map.removeLayer(layer));
          pathLayers = [];
        }

        // Show legend
        document.getElementById('algorithm-legend').style.display = 'flex';

        // Process each algorithm
        const results = [];
        let completed = 0;

        selectedAlgorithms.forEach(algorithm => {
          const formData = new FormData();
          formData.append('start_lat', startMarker.getLatLng().lat);
          formData.append('start_lng', startMarker.getLatLng().lng);
          formData.append('end_lat', endMarker.getLatLng().lat);
          formData.append('end_lng', endMarker.getLatLng().lng);
          formData.append('algorithm', algorithm);

          fetch('/find_path', {
            method: 'POST',
            body: formData
          })
          .then(response => response.json())
          .then(data => {
            completed++;
            if (data.success) {
              const colors = {
                'astar': '#667eea',
                'dijkstra': '#4facfe',
                'ucs': '#43e97b',
                'bfs': '#fa709a'
              };
              const pathColor = colors[algorithm] || '#ff0000';
              const polyline = L.polyline(data.path, { 
                color: pathColor, 
                weight: 5, 
                opacity: 0.8 
              }).addTo(map);
              pathLayers.push(polyline);

              // Add label for the path
              const midpoint = Math.floor(data.path.length / 2);
              const labelPos = data.path[midpoint] || data.path[0];
              const marker = L.marker(labelPos, {
                icon: L.divIcon({
                  className: 'algorithm-label',
                  html: `<div style="background: ${pathColor}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; box-shadow: 0 2px 4px rgba(0,0,0,0.3);">${algorithm.toUpperCase()}: ${data.length}</div>`,
                  iconSize: [100, 30],
                  iconAnchor: [50, 15]
                })
              }).addTo(map);
              pathLayers.push(marker);

              results.push(`${algorithm.toUpperCase()}: ${data.length}`);
            }
            if (completed === selectedAlgorithms.length) {
              findPathBtn.disabled = false;
              statusDiv.textContent = `Found ${completed} path(s). Compare the results!`;
              pathInfoDiv.textContent = results.join(' | ');
              // Fit map to show all paths
              if (pathLayers.length > 0) {
                const group = L.featureGroup(pathLayers.filter(l => l instanceof L.Polyline));
                map.fitBounds(group.getBounds(), { padding: [50, 50] });
              }
            }
          })
          .catch(error => {
            completed++;
            console.error('Error:', error);
            if (completed === selectedAlgorithms.length) {
              findPathBtn.disabled = false;
              statusDiv.textContent = 'Some paths failed to calculate.';
            }
          });
        });
      });

      resetBtn.addEventListener('click', resetPoints);

      function resetPoints() {
        if (startMarker) { map.removeLayer(startMarker); startMarker = null; }
        if (endMarker) { map.removeLayer(endMarker); endMarker = null; }
        if (pathLayers) {
          pathLayers.forEach(layer => map.removeLayer(layer));
          pathLayers = [];
        }
        document.getElementById('algorithm-legend').style.display = 'none';
        findPathBtn.disabled = true;
        pathInfoDiv.textContent = '';
        if (isRegionLoaded) {
          statusDiv.textContent = 'Markers reset. Click on the map to select new points.';
        }
      }
    });
  </script>
</body>
</html>
'''

@app.route('/')
def index():
    regions = path_finder.get_available_regions()
    return render_template_string(BASE_TEMPLATE, regions=regions)

@app.route('/load_region', methods=['POST'])
def load_region():
    region_name = request.form.get('region')
    success = path_finder.load_region(region_name)
    if success:
        bounds = path_finder.get_region_bounds()
        nodes_list = path_finder.get_all_nodes()
        return jsonify({
            'success': True,
            'bounds': bounds,
            'nodes': nodes_list,
            'message': f"{region_name} loaded successfully."
        })
    else:
        return jsonify({
            'success': False,
            'message': f"Failed to load {region_name}. Please try a different region."
        })

@app.route('/find_path', methods=['POST'])
def find_path():
    try:
        start_lat = float(request.form.get('start_lat'))
        start_lng = float(request.form.get('start_lng'))
        end_lat = float(request.form.get('end_lat'))
        end_lng = float(request.form.get('end_lng'))
        algorithm = request.form.get('algorithm', 'astar')
        print(f"Finding path from ({start_lat}, {start_lng}) to ({end_lat}, {end_lng}) using {algorithm}")

        start_node = path_finder.find_nearest_node(start_lat, start_lng)
        end_node = path_finder.find_nearest_node(end_lat, end_lng)

        if start_node is None or end_node is None:
            return jsonify({'success': False, 'message': 'Could not find nodes near your points.'})
        if start_node == end_node:
            return jsonify({'success': False, 'message': 'Selected points are too close together.'})

        path_coords, path_length = path_finder.find_shortest_path(start_node, end_node, algorithm)
        if not path_coords or len(path_coords) < 2:
            return jsonify({'success': False, 'message': 'No valid path found.'})

        return jsonify({
            'success': True,
            'path': path_coords,
            'length': f"{path_length/1000:.2f} km",
            'algorithm': algorithm.upper(),
            'message': f'Path found using {algorithm.upper()}.'
        })
    except Exception as e:
        print(f"Error in find_path: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'Error finding path. Please try again.'})

if __name__ == '__main__':
    app.run(debug=True)
