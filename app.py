# Flask Web Application - Fashion AI Assistant
from flask import Flask, render_template, request, jsonify
from Agent import Agent
import logging
import requests
import base64
import json
import os
from urllib.parse import urlparse
import hashlib
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app and AI agent
app = Flask(__name__)
agent = Agent()

# Create directory for cached product images
IMAGES_CACHE_DIR = "static/images/cache"
os.makedirs(IMAGES_CACHE_DIR, exist_ok=True)

def download_and_cache_image(image_url, max_retries=2):
    """Download product images from Zara and cache them locally"""
    if not image_url or image_url == "":
        return None
    
    try:
        # Create unique filename from URL hash
        url_hash = hashlib.md5(image_url.encode()).hexdigest()
        parsed_url = urlparse(image_url)
        file_extension = os.path.splitext(parsed_url.path)[1] or '.jpg'
        cached_filename = f"{url_hash}{file_extension}"
        cached_path = os.path.join(IMAGES_CACHE_DIR, cached_filename)
        relative_path = f"/static/images/cache/{cached_filename}"
        
        # Return cached image if it already exists
        if os.path.exists(cached_path):
            logger.info(f"Image already cached: {cached_filename}")
            return relative_path
        
        # Try downloading with retries and different user agents
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"Downloading image from: {image_url} (attempt {attempt + 1})")
                
                # Rotate through different browser user agents
                user_agents = [
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
                ]
                
                # Set browser-like headers to avoid blocking
                headers = {
                    'User-Agent': user_agents[attempt % len(user_agents)],
                    'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                }
                
                # Increase timeout with each retry
                timeout = 10 + (attempt * 5)
                
                response = requests.get(
                    image_url, 
                    timeout=timeout, 
                    headers=headers,
                    allow_redirects=True,
                    stream=True
                )
                response.raise_for_status()
                
                # Verify response is actually an image
                content_type = response.headers.get('content-type', '').lower()
                if not any(img_type in content_type for img_type in ['image/', 'jpeg', 'jpg', 'png', 'webp']):
                    logger.warning(f"Response is not an image: {content_type}")
                    continue
                
                # Save image to cache directory
                with open(cached_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Verify file was saved successfully
                if os.path.exists(cached_path) and os.path.getsize(cached_path) > 0:
                    logger.info(f"Image cached successfully: {cached_filename}")
                    return relative_path
                else:
                    logger.warning(f"Cached file is empty or doesn't exist")
                    if os.path.exists(cached_path):
                        os.remove(cached_path)
                    continue
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout downloading image (attempt {attempt + 1}): {image_url}")
                time.sleep(1)  # Brief pause before retry
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error downloading image (attempt {attempt + 1}): {image_url}")
                time.sleep(1)
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    logger.warning(f"Image not found (404): {image_url}")
                    return None  # Don't retry for 404s
                else:
                    logger.warning(f"HTTP error {e.response.status_code} (attempt {attempt + 1}): {image_url}")
                    time.sleep(1)
            except Exception as e:
                logger.warning(f"Unexpected error downloading image (attempt {attempt + 1}): {e}")
                time.sleep(1)
        
        logger.error(f"Failed to download image after {max_retries + 1} attempts: {image_url}")
        return None
        
    except Exception as e:
        logger.error(f"Error in download_and_cache_image: {e}")
        return None

def process_product_images(products):
    """Download and cache all product images in parallel"""
    if not products:
        return products
    
    total_images = len(products)
    successful_downloads = 0
    
    for i, product in enumerate(products):
        logger.info(f"Processing image {i+1}/{total_images}")
        
        if 'image_url' in product and product['image_url']:
            # Download and cache the product image
            cached_path = download_and_cache_image(product['image_url'])
            product['cached_image_path'] = cached_path
            product['image_available'] = cached_path is not None
            
            if cached_path:
                successful_downloads += 1
        else:
            product['cached_image_path'] = None
            product['image_available'] = False
    
    logger.info(f"Image processing complete: {successful_downloads}/{total_images} images successfully cached")
    return products

@app.route('/')
def index():
    """Serve the main chat interface"""
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle user messages and return fashion recommendations"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        logger.info(f"Processing user message: {user_message}")
        
        # Process message through AI agent
        result = agent.process_user_request(user_message)
        
        # Handle conversational responses (greetings, thanks, etc.)
        if result.get('message_type') == 'conversational':
            return jsonify({
                'status': 'success',
                'message_type': 'conversational',
                'response': result['response'],
                'user_query': user_message
            })
        
        # Handle system explanation requests
        elif result.get('message_type') == 'explanation':
            return jsonify({
                'status': 'success',
                'message_type': 'explanation',
                'response': result['response'],
                'user_query': user_message
            })
        
        # Handle fashion recommendations with products
        elif result['status'] == 'success' and 'recommendations' in result:
            recommendations = result['recommendations']
            
            # Download and cache product images
            if 'products' in recommendations and recommendations['products']:
                logger.info(f"Starting image download for {len(recommendations['products'])} products")
                recommendations['products'] = process_product_images(recommendations['products'])
            
            # Calculate outfit cost metrics
            if 'products' in recommendations and recommendations['products']:
                prices = []
                for product in recommendations['products']:
                    try:
                        # Extract numeric price from "$49.99" format
                        price_str = product.get('price', '0')
                        price_clean = price_str.replace('$', '').replace(',', '')
                        price = float(price_clean)
                        prices.append(price)
                    except (ValueError, AttributeError):
                        continue
                
                # Add cost summary to response
                if prices:
                    total_cost = sum(prices)
                    avg_price = total_cost / len(prices)
                    images_loaded = sum(1 for product in recommendations['products'] if product.get('image_available', False))
                    
                    recommendations['outfit_metrics'] = {
                        'total_price': f"${total_cost:.2f}",
                        'item_count': len(prices),
                        'average_price': f"${avg_price:.2f}",
                        'price_range': f"${min(prices):.2f} - ${max(prices):.2f}",
                        'images_loaded': images_loaded
                    }
            
            return jsonify({
                'status': 'success',
                'message_type': 'structured_outfit',
                'recommendations': recommendations,
                'style_analysis': result.get('style_analysis', ''),
                'user_query': user_message
            })
        
        # Handle clarification requests
        elif result.get('message_type') == 'clarification':
            return jsonify({
                'status': 'success',
                'message_type': 'clarification',
                'response': result['response'],
                'user_query': user_message
            })
        
        # Handle errors or unknown responses
        else:
            return jsonify({
                'status': 'error',
                'message': result.get('message', 'Something went wrong. Please try again!'),
                'user_query': user_message
            })
            
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Oops! Something went wrong. Please try again in a moment! 😊'
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'service': 'fashion-agent',
        'version': '1.0'
    })

@app.route('/clear-cache', methods=['POST'])
def clear_image_cache():
    """Clear cached product images (for maintenance)"""
    try:
        cache_files = os.listdir(IMAGES_CACHE_DIR)
        for file in cache_files:
            file_path = os.path.join(IMAGES_CACHE_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        return jsonify({
            'status': 'success',
            'message': f'Cleared {len(cache_files)} cached images'
        })
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to clear cache'
        }), 500

if __name__ == '__main__':
    logger.info("Starting Enhanced Fashion Agent Flask Application with RAG capabilities")
    logger.info(f"Image cache directory: {IMAGES_CACHE_DIR}")
    logger.info("Features: Intent Recognition, Conversational AI, Image Caching, RAG-based Recommendations")
    
    # Run the Flask development server
    app.run(debug=True, host='0.0.0.0', port=5000) 