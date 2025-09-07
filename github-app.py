# github_app.py - Simplified for GitHub Actions
import json
import base64
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from scipy import ndimage
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_area(data):
    try:
        coordinates = data.get('coordinates')
        alteration_params = data.get('parameters', {})

        if not coordinates:
            return {
                'status': 'error',
                'message': 'No coordinates provided'
            }

        # Convert coordinates to float
        coords_array = coordinates.split(',')
        if len(coords_array) != 4:
            return {
                'status': 'error',
                'message': 'Invalid coordinates format. Expected 4 values separated by commas.'
            }
            
        nw_lat, nw_lon, se_lat, se_lon = [float(coord.strip()) for coord in coords_array]

        # Create simulated data
        results = create_simulated_data()
        data_source = "simulated data"
        
        exploration_map = create_exploration_target(results, alteration_params)
        img_buffer = create_output_image(exploration_map, nw_lat, nw_lon, se_lat, se_lon)

        return {
            'status': 'success',
            'image': base64.b64encode(img_buffer.getvalue()).decode('utf-8'),
            'bounds': [nw_lat, nw_lon, se_lat, se_lon],
            'message': f'Processing completed successfully using {data_source}',
            'products_found': 3  # Simulated count
        }

    except Exception as e:
        logger.error(f"Error in process_area: {e}")
        return {
            'status': 'error',
            'message': str(e)
        }

def create_simulated_data():
    """Create simulated geologic data"""
    width, height = 100, 100

    # Create random but structured data
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    
    # Create different alteration patterns
    clay_ratio = np.sin(5*x) * np.cos(5*y) + 0.5
    iron_ratio = np.exp(-((x-0.5)**2 + (y-0.5)**2)/0.1) + 0.3
    silica_ratio = np.abs(np.sin(3*x) * np.cos(4*y)) + 0.4
    carbonate_ratio = np.cos(6*x) * np.sin(6*y) + 0.5

    # Extract lineaments
    lineaments = simple_edge_detection(clay_ratio)
    line_density = calculate_line_density(lineaments)

    return {
        'clay_ratio': clay_ratio,
        'iron_ratio': iron_ratio,
        'silica_ratio': silica_ratio,
        'carbonate_ratio': carbonate_ratio,
        'line_density': line_density
    }

def calculate_band_ratios(band1, band2):
    return np.divide(band1.astype(float), band2.astype(float),
                    out=np.zeros_like(band1.astype(float)),
                    where=band2 != 0)

def simple_edge_detection(data, threshold=0.1):
    sx = ndimage.sobel(data, axis=0, mode='constant')
    sy = ndimage.sobel(data, axis=1, mode='constant')
    sob = np.hypot(sx, sy)
    return sob > threshold * np.max(sob)

def calculate_line_density(lineaments, window_size=15):
    kernel = np.ones((window_size, window_size))
    density = ndimage.convolve(lineaments.astype(float), kernel) / (window_size ** 2)
    return density

def create_exploration_target(results, alteration_params):
    weighted_sum = np.zeros_like(results['clay_ratio'])
    total_weight = 0

    alteration_types = ['clay', 'iron', 'silica', 'carbonate', 'line_density']

    for alt_type in alteration_types:
        include_key = f"include_{alt_type}"
        threshold_key = f"threshold_{alt_type}"
        weight_key = f"weight_{alt_type}"

        if alt_type in alteration_params and alteration_params[include_key]:
            threshold = float(alteration_params.get(threshold_key, 80))
            weight = float(alteration_params.get(weight_key, 1.0))

            threshold_value = np.percentile(results[f"{alt_type}_ratio"], threshold)
            binary_map = results[f"{alt_type}_ratio"] > threshold_value

            weighted_sum += binary_map.astype(float) * weight
            total_weight += weight

    if total_weight > 0:
        weighted_sum /= total_weight

    exploration_map = np.zeros_like(weighted_sum, dtype=np.uint8)
    exploration_map[weighted_sum < 0.3] = 1
    exploration_map[(weighted_sum >= 0.3) & (weighted_sum < 0.6)] = 2
    exploration_map[(weighted_sum >= 0.6) & (weighted_sum < 0.8)] = 3
    exploration_map[weighted_sum >= 0.8] = 4

    return exploration_map

def create_output_image(exploration_map, nw_lat, nw_lon, se_lat, se_lon):
    """Create PNG image from exploration map"""
    height, width = exploration_map.shape

    cmap = plt.cm.colors.ListedColormap(['grey', 'green', 'orange', 'red'])
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(exploration_map, cmap=cmap, norm=norm, interpolation='nearest')
    
    cbar = plt.colorbar(ticks=[1, 2, 3, 4])
    cbar.ax.set_yticklabels(['Non-prospective', 'Low prospective', 'Moderate prospective', 'High prospective'])
    
    plt.title('Geologic Exploration Results')
    plt.axis('off')

    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
    plt.close()

    img_buffer.seek(0)
    return img_buffer

# HTTP handler for GitHub Actions
class Handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_POST(self):
        if self.path == '/process':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                result = process_area(data)
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                self.wfile.write(json.dumps(result).encode('utf-8'))
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                error_response = {
                    'status': 'error',
                    'message': f'Internal server error: {str(e)}'
                }
                self.wfile.write(json.dumps(error_response).encode('utf-8'))
        else:
            self.send_response(404)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            error_response = {
                'status': 'error',
                'message': 'Endpoint not found'
            }
            self.wfile.write(json.dumps(error_response).encode('utf-8'))

def run_server():
    server = HTTPServer(('localhost', 8080), Handler)
    print('Starting server on port 8080...')
    server.serve_forever()

if __name__ == '__main__':
    # Start the server
    run_server()
