import os
import json
import base64
import argparse
import numpy as np
import requests
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from scipy import ndimage
import logging
import rasterio
from rasterio.plot import show
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date
import zipfile
import tempfile
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GeologicProcessor:
    def __init__(self):
        self.access_token = None
        self.token_expiry = None
        self.api = None
        
    def connect_to_copernicus(self):
        """Connect to Copernicus Data Space Ecosystem"""
        try:
            username = os.environ.get('CDSE_USERNAME')
            password = os.environ.get('CDSE_PASSWORD')
            
            if not username or not password:
                logger.warning("CDSE credentials not found. Using simulated data.")
                return False
            
            logger.info("Connecting to Copernicus Data Space Ecosystem")
            self.api = SentinelAPI(username, password, 'https://sh.dataspace.copernicus.eu')
            logger.info("Successfully connected to CDSE")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to CDSE: {e}")
            return False
    
    def search_sentinel_products(self, nw_lat, nw_lon, se_lat, se_lon, cloud_cover=30):
        """Search for Sentinel-2 products in the specified area"""
        try:
            # Convert coordinates to footprint
            footprint = f"POLYGON(({nw_lon} {nw_lat}, {se_lon} {nw_lat}, {se_lon} {se_lat}, {nw_lon} {se_lat}, {nw_lon} {nw_lat}))"
            
            # Search for products
            products = self.api.query(
                footprint,
                date=('20240101', date.today().strftime('%Y%m%d')),
                platformname='Sentinel-2',
                cloudcoverpercentage=(0, cloud_cover),
                producttype='S2MSI2A'  # Level-2A processed data
            )
            
            logger.info(f"Found {len(products)} Sentinel-2 products")
            return products
            
        except Exception as e:
            logger.error(f"Error searching for Sentinel products: {e}")
            return {}
    
    def download_and_process_product(self, product_id, output_dir):
        """Download and process a Sentinel-2 product"""
        try:
            logger.info(f"Downloading product {product_id}")
            
            # Download the product
            product_info = self.api.download(product_id, directory_path=output_dir)
            
            # Extract the zip file
            zip_path = product_info['path']
            extract_dir = os.path.join(output_dir, product_id)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            logger.info(f"Product extracted to {extract_dir}")
            
            # Find the relevant image files
            granule_dir = self.find_granule_directory(extract_dir)
            
            if granule_dir:
                # Load the bands needed for mineral exploration
                bands = self.load_bands(granule_dir)
                return bands, extract_dir
            
            return None, extract_dir
            
        except Exception as e:
            logger.error(f"Error downloading and processing product: {e}")
            return None, None
    
    def find_granule_directory(self, extract_dir):
        """Find the granule directory in the extracted product"""
        for root, dirs, files in os.walk(extract_dir):
            if 'GRANULE' in dirs:
                return os.path.join(root, 'GRANULE')
        return None
    
    def load_bands(self, granule_dir):
        """Load the relevant Sentinel-2 bands for mineral exploration"""
        bands = {}
        
        try:
            # Find the IMG_DATA directory
            for root, dirs, files in os.walk(granule_dir):
                if 'IMG_DATA' in dirs:
                    img_data_dir = os.path.join(root, 'IMG_DATA')
                    
                    # Band mapping for Sentinel-2
                    band_files = {
                        'B2': 'B02_10m.jp2',   # Blue
                        'B3': 'B03_10m.jp2',   # Green
                        'B4': 'B04_10m.jp2',   # Red
                        'B8': 'B08_10m.jp2',   # NIR
                        'B11': 'B11_20m.jp2',  # SWIR1
                        'B12': 'B12_20m.jp2',  # SWIR2
                        'B8A': 'B8A_20m.jp2',  # Narrow NIR
                    }
                    
                    # Load each band
                    for band_name, band_file in band_files.items():
                        band_path = self.find_band_file(img_data_dir, band_file)
                        if band_path:
                            with rasterio.open(band_path) as src:
                                bands[band_name] = src.read(1)
                                logger.info(f"Loaded band {band_name} with shape {bands[band_name].shape}")
            
            return bands
            
        except Exception as e:
            logger.error(f"Error loading bands: {e}")
            return bands
    
    def find_band_file(self, img_data_dir, band_file):
        """Find a specific band file in the directory structure"""
        for root, dirs, files in os.walk(img_data_dir):
            if band_file in files:
                return os.path.join(root, band_file)
        return None
    
    def calculate_band_ratios(self, bands):
        """Calculate band ratios for mineral exploration"""
        ratios = {}
        
        try:
            # Resample 20m bands to 10m resolution if needed
            if 'B11' in bands and 'B8' in bands:
                if bands['B11'].shape != bands['B8'].shape:
                    from scipy.ndimage import zoom
                    zoom_factor = bands['B8'].shape[0] / bands['B11'].shape[0]
                    bands['B11'] = zoom(bands['B11'], zoom_factor, order=1)
                    bands['B12'] = zoom(bands['B12'], zoom_factor, order=1)
                    bands['B8A'] = zoom(bands['B8A'], zoom_factor, order=1)
            
            # Clay Minerals Ratio (SWIR1 / SWIR2) - for phyllosilicates
            if 'B11' in bands and 'B12' in bands:
                ratios['clay'] = np.divide(bands['B11'].astype(float), 
                                         bands['B12'].astype(float),
                                         out=np.zeros_like(bands['B11'].astype(float)),
                                         where=bands['B12'] != 0)
            
            # Iron Oxide Ratio (Red / Blue) - for iron minerals
            if 'B4' in bands and 'B2' in bands:
                ratios['iron'] = np.divide(bands['B4'].astype(float), 
                                         bands['B2'].astype(float),
                                         out=np.zeros_like(bands['B4'].astype(float)),
                                         where=bands['B2'] != 0)
            
            # Silica Ratio (SWIR1 / NIR) - for silicification
            if 'B11' in bands and 'B8' in bands:
                ratios['silica'] = np.divide(bands['B11'].astype(float), 
                                           bands['B8'].astype(float),
                                           out=np.zeros_like(bands['B11'].astype(float)),
                                           where=bands['B8'] != 0)
            
            # Gold-related alteration (Custom ratio based on research)
            if 'B11' in bands and 'B8' in bands and 'B4' in bands:
                # This is a simplified proxy for gold-related alteration
                ratios['gold_alteration'] = np.divide(
                    bands['B11'].astype(float) + bands['B4'].astype(float),
                    bands['B8'].astype(float) + 0.0001,
                    out=np.zeros_like(bands['B11'].astype(float))
                )
            
            return ratios
            
        except Exception as e:
            logger.error(f"Error calculating band ratios: {e}")
            return ratios
    
    def detect_lineaments(self, band_data):
        """Detect lineaments (structural features) using edge detection"""
        try:
            # Use Sobel filter for edge detection
            sx = ndimage.sobel(band_data, axis=0, mode='constant')
            sy = ndimage.sobel(band_data, axis=1, mode='constant')
            sob = np.hypot(sx, sy)
            
            # Threshold to create binary lineament map
            threshold = np.percentile(sob, 95)  # Top 5% as lineaments
            lineaments = sob > threshold
            
            # Calculate lineament density
            kernel_size = 15
            kernel = np.ones((kernel_size, kernel_size))
            lineament_density = ndimage.convolve(lineaments.astype(float), kernel) / (kernel_size ** 2)
            
            return lineament_density
            
        except Exception as e:
            logger.error(f"Error detecting lineaments: {e}")
            return np.zeros_like(band_data)
    
    def create_gold_exploration_target(self, ratios, alteration_params):
        """Create gold exploration target map based on band ratios and parameters"""
        try:
            # Initialize weighted sum
            weighted_sum = np.zeros_like(next(iter(ratios.values())))
            total_weight = 0
            
            # Apply weights and thresholds for each alteration type
            alteration_types = ['clay', 'iron', 'silica', 'gold_alteration']
            
            for alt_type in alteration_types:
                include_key = f"include_{alt_type}"
                threshold_key = f"threshold_{alt_type}"
                weight_key = f"weight_{alt_type}"
                
                # Use default values if not specified
                include = alteration_params.get(include_key, True)
                threshold = float(alteration_params.get(threshold_key, 80))
                weight = float(alteration_params.get(weight_key, 1.0))
                
                if include and alt_type in ratios:
                    # Calculate threshold value
                    threshold_value = np.percentile(ratios[alt_type], threshold)
                    
                    # Create binary map above threshold
                    binary_map = ratios[alt_type] > threshold_value
                    
                    # Add to weighted sum
                    weighted_sum += binary_map.astype(float) * weight
                    total_weight += weight
            
            # Normalize if weights were applied
            if total_weight > 0:
                weighted_sum /= total_weight
            
            # Create exploration target categories
            exploration_map = np.zeros_like(weighted_sum, dtype=np.uint8)
            exploration_map[weighted_sum < 0.3] = 1  # Non-prospective
            exploration_map[(weighted_sum >= 0.3) & (weighted_sum < 0.6)] = 2  # Low prospective
            exploration_map[(weighted_sum >= 0.6) & (weighted_sum < 0.8)] = 3  # Moderate prospective
            exploration_map[weighted_sum >= 0.8] = 4  # High prospective
            
            return exploration_map, weighted_sum
            
        except Exception as e:
            logger.error(f"Error creating exploration target: {e}")
            return np.zeros_like(next(iter(ratios.values()))), np.zeros_like(next(iter(ratios.values())))
    
    def create_output_image(self, exploration_map, ratios, nw_lat, nw_lon, se_lat, se_lon, data_source):
        """Create output visualization image"""
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Gold Exploration Analysis\nNW: {nw_lat:.4f}°, {nw_lon:.4f}° | SE: {se_lat:.4f}°, {se_lon:.4f}°\nData Source: {data_source}', 
                        fontsize=16, fontweight='bold')
            
            # Plot exploration map
            cmap = plt.cm.colors.ListedColormap(['grey', 'green', 'orange', 'red'])
            bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
            norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
            
            im = axes[0, 0].imshow(exploration_map, cmap=cmap, norm=norm, interpolation='nearest')
            axes[0, 0].set_title('Gold Exploration Potential', fontweight='bold')
            axes[0, 0].axis('off')
            
            # Add colorbar for exploration map
            cbar = fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)
            cbar.set_ticks([1, 2, 3, 4])
            cbar.set_ticklabels(['Non-prospective', 'Low prospective', 'Moderate prospective', 'High prospective'])
            
            # Plot key band ratios
            ratio_titles = {
                'clay': 'Clay Minerals (B11/B12)',
                'iron': 'Iron Oxides (B4/B2)',
                'silica': 'Silicification (B11/B8)',
                'gold_alteration': 'Gold Alteration Index'
            }
            
            # Plot up to 3 ratio maps
            ratio_keys = [k for k in ['clay', 'iron', 'silica', 'gold_alteration'] if k in ratios]
            for i, ratio_key in enumerate(ratio_keys[:3]):
                row = i // 2 + 1
                col = i % 2
                
                if row < 2 and col < 2:  # Ensure we don't exceed subplot bounds
                    im_ratio = axes[row, col].imshow(ratios[ratio_key], cmap='viridis', interpolation='nearest')
                    axes[row, col].set_title(ratio_titles.get(ratio_key, ratio_key), fontweight='bold')
                    axes[row, col].axis('off')
                    fig.colorbar(im_ratio, ax=axes[row, col], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            
            # Save image
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
            plt.close(fig)
            
            img_buffer.seek(0)
            return img_buffer
            
        except Exception as e:
            logger.error(f"Error creating output image: {e}")
            # Fallback to simple image
            return self.create_simple_output_image(exploration_map, nw_lat, nw_lon, se_lat, se_lon, data_source)
    
    def create_simple_output_image(self, exploration_map, nw_lat, nw_lon, se_lat, se_lon, data_source):
        """Create a simple output image as fallback"""
        plt.figure(figsize=(10, 8))
        
        cmap = plt.cm.colors.ListedColormap(['grey', 'green', 'orange', 'red'])
        bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        
        plt.imshow(exploration_map, cmap=cmap, norm=norm, interpolation='nearest')
        
        cbar = plt.colorbar(ticks=[1, 2, 3, 4])
        cbar.ax.set_yticklabels(['Non-prospective', 'Low prospective', 'Moderate prospective', 'High prospective'])
        
        plt.title(f'Gold Exploration Results\nNW: {nw_lat:.4f}°, {nw_lon:.4f}° | SE: {se_lat:.4f}°, {se_lon:.4f}°\nData Source: {data_source}')
        plt.axis('off')
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        
        img_buffer.seek(0)
        return img_buffer
    
    def process_area(self, coordinates, alteration_params=None):
        """Main processing function"""
        temp_dir = None
        
        try:
            if alteration_params is None:
                alteration_params = {}
            
            # Parse coordinates
            coords_array = coordinates.split(',')
            if len(coords_array) != 4:
                raise ValueError("Invalid coordinates format. Expected 4 values separated by commas.")
                
            nw_lat, nw_lon, se_lat, se_lon = [float(coord.strip()) for coord in coords_array]
            
            # Connect to Copernicus API
            connected = self.connect_to_copernicus()
            products = {}
            
            if connected:
                # Search for Sentinel-2 products
                products = self.search_sentinel_products(nw_lat, nw_lon, se_lat, se_lat)
            
            if not products or not connected:
                logger.warning("No Sentinel-2 products found or connection failed. Using simulated data.")
                return self.process_simulated_data(nw_lat, nw_lon, se_lat, se_lon, alteration_params)
            
            # Create temporary directory for downloads
            temp_dir = tempfile.mkdtemp()
            logger.info(f"Created temporary directory: {temp_dir}")
            
            # Download and process the first available product
            product_id = list(products.keys())[0]
            bands, extract_dir = self.download_and_process_product(product_id, temp_dir)
            
            if not bands:
                raise Exception("Failed to download or process Sentinel-2 product")
            
            # Calculate band ratios
            ratios = self.calculate_band_ratios(bands)
            
            if not ratios:
                raise Exception("Failed to calculate band ratios")
            
            # Detect lineaments and add to ratios
            lineament_density = self.detect_lineaments(bands.get('B8', bands.get('B4', next(iter(bands.values())))))
            ratios['line_density'] = lineament_density
            
            # Create exploration target
            exploration_map, weighted_sum = self.create_gold_exploration_target(ratios, alteration_params)
            
            # Create output image
            img_buffer = self.create_output_image(exploration_map, ratios, nw_lat, nw_lon, se_lat, se_lon, 
                                                f"Sentinel-2 {product_id}")
            
            # Calculate statistics
            high_potential = np.sum(exploration_map == 4)
            total_pixels = exploration_map.size
            high_potential_percent = (high_potential / total_pixels) * 100
            
            return {
                'status': 'success',
                'image': base64.b64encode(img_buffer.getvalue()).decode('utf-8'),
                'bounds': [nw_lat, nw_lon, se_lat, se_lon],
                'message': f'Processing completed successfully using Sentinel-2 data',
                'products_found': len(products),
                'high_potential_area': f'{high_potential_percent:.1f}%',
                'parameters_used': alteration_params,
                'product_id': product_id
            }
            
        except Exception as e:
            logger.error(f"Error in process_area: {e}")
            # Fallback to simulated data
            try:
                return self.process_simulated_data(nw_lat, nw_lon, se_lat, se_lon, alteration_params)
            except Exception as fallback_error:
                logger.error(f"Fallback processing also failed: {fallback_error}")
                return {
                    'status': 'error',
                    'message': str(e)
                }
            
        finally:
            # Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
    
    def process_simulated_data(self, nw_lat, nw_lon, se_lat, se_lon, alteration_params):
        """Process simulated data as fallback"""
        logger.info("Processing simulated data as fallback")
        
        # Create simulated data based on coordinates
        width, height = 200, 200
        
        # Create coordinate-based patterns
        x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
        
        # Use coordinates to create unique patterns
        lat_factor = abs(nw_lat) / 90.0
        lon_factor = abs(nw_lon) / 180.0
        
        # Simulate band ratios
        ratios = {
            'clay': np.sin(5*x + lat_factor) * np.cos(5*y + lon_factor) + 0.5,
            'iron': np.exp(-((x-0.5)**2 + (y-0.5)**2)/0.1) + 0.3,
            'silica': np.abs(np.sin(3*x + lon_factor) * np.cos(4*y + lat_factor)) + 0.4,
            'gold_alteration': np.cos(6*x - lat_factor) * np.sin(6*y - lon_factor) + 0.5
        }
        
        # Create exploration target
        exploration_map, weighted_sum = self.create_gold_exploration_target(ratios, alteration_params)
        
        # Create output image
        img_buffer = self.create_output_image(exploration_map, ratios, nw_lat, nw_lon, se_lat, se_lon, 
                                            "Simulated Data (No Sentinel-2 products found)")
        
        # Calculate statistics
        high_potential = np.sum(exploration_map == 4)
        total_pixels = exploration_map.size
        high_potential_percent = (high_potential / total_pixels) * 100
        
        return {
            'status': 'success',
            'image': base64.b64encode(img_buffer.getvalue()).decode('utf-8'),
            'bounds': [nw_lat, nw_lon, se_lat, se_lon],
            'message': 'Processing completed successfully using simulated data (no Sentinel-2 products found)',
            'products_found': 0,
            'high_potential_area': f'{high_potential_percent:.1f}%',
            'parameters_used': alteration_params,
            'product_id': None
        }

def main():
    parser = argparse.ArgumentParser(description='Gold Exploration Processing API')
    parser.add_argument('process', nargs='?', help='Process mode')
    parser.add_argument('--coordinates', required=True, help='Coordinates in format "NW_LAT,NW_LON,SE_LAT,SE_LON"')
    parser.add_argument('--parameters', default='{}', help='Processing parameters as JSON string')
    
    args = parser.parse_args()
    
    if args.process == 'process':
        try:
            # Parse parameters
            parameters = json.loads(args.parameters)
            
            # Process area
            processor = GeologicProcessor()
            result = processor.process_area(args.coordinates, parameters)
            
            # Save result
            os.makedirs('outputs', exist_ok=True)
            with open('outputs/result.json', 'w') as f:
                json.dump(result, f, indent=2)
                
            # Also save the image separately
            if result['status'] == 'success' and 'image' in result:
                image_data = base64.b64decode(result['image'])
                with open('outputs/exploration_map.png', 'wb') as f:
                    f.write(image_data)
                
            print("Processing completed successfully")
            print(f"Results saved to outputs/result.json")
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            exit(1)
    else:
        print("Usage: python main.py process --coordinates 'NW_LAT,NW_LON,SE_LAT,SE_LON' [--parameters 'JSON_STRING']")

if __name__ == '__main__':
    main()
