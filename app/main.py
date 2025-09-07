# app/main.py - GitHub Actions version
import os
import json
import base64
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from scipy import ndimage
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health_check():
    logger.info("Health check endpoint called")
    return jsonify({"status": "healthy", "message": "Geologic processing API is running"})

@app.route('/process', methods=['POST'])
def process_area():
    try:
        logger.info("Process endpoint called")
        data = request.get_json()
        if not data:
            logger.error("No JSON data received")
            return jsonify({
                'status': 'error',
                'message': 'No JSON data received'
            }), 400
            
        coordinates = data.get('coordinates')
        alteration_params = data.get('parameters', {})

        if not coordinates:
            logger.error("No coordinates provided")
            return jsonify({
                'status': 'error',
                'message': 'No coordinates provided'
            }), 400

        # Convert coordinates to float
        coords_array = coordinates.split(',')
        if len(coords_array) != 4:
            logger.error(f"Invalid coordinates format: {coordinates}")
            return jsonify({
                'status': 'error',
                'message': 'Invalid coordinates format. Expected 4 values separated by commas.'
            }), 400
            
        nw_lat, nw_lon, se_lat, se_lon = [float(coord.strip()) for coord in coords_array]
        logger.info(f"Processing coordinates: {nw_lat}, {nw_lon}, {se_lat}, {se_lon}")

        # Process the data (simulated for now)
        results = process_geologic_data(nw_lat, nw_lon, se_lat, se_lon)
        exploration_map = create_exploration_target(results, alteration_params)
        img_buffer = create_output_image(exploration_map, nw_lat, nw_lon, se_lat, se_lon)

        logger.info("Processing completed successfully")
        return jsonify({
            'status': 'success',
            'image': base64.b64encode(img_buffer.getvalue()).decode('utf-8'),
            'bounds': [nw_lat, nw_lon, se_lat, se_lon],
            'message': 'Processing completed successfully using simulated data',
            'products_found': 3,
            'parameters_used': alteration_params
        })

    except Exception as e:
        logger.error(f"Error in process_area: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def process_geologic_data(nw_lat, nw_lon, se_lat, se_lon):
    """Process geologic data - simulated version"""
    logger.info(f"Processing area: {nw_lat}, {nw_lon}, {se_lat}, {se_lon}")
    
    # Create realistic simulated data based on coordinates
    width, height = 200, 200
    
    # Create coordinate-based patterns
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    
    # Create different alteration patterns based on coordinates
    clay_ratio = np.sin(5*x + nw_lat/10) * np.cos(5*y + nw_lon/10) + 0.5
    iron_ratio = np.exp(-((x-0.5)**2 + (y-0.5)**2)/0.1) + 0.3 + (nw_lat % 10)/100
    silica_ratio = np.abs(np.sin(3*x + nw_lon/20) * np.cos(4*y + nw_lat/20)) + 0.4
    carbonate_ratio = np.cos(6*x - nw_lat/15) * np.sin(6*y - nw_lon/15) + 0.5

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
    
    plt.title(f'Geologic Exploration Results\nNW: {nw_lat:.4f}°, {nw_lon:.4f}° | SE: {se_lat:.4f}°, {se_lon:.4f}°')
    plt.axis('off')

    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
    plt.close()

    img_buffer.seek(0)
    return img_buffer

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting server on port {port}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Files in directory: {os.listdir('.')}")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}", exc_info=True)
        raise
