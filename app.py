from flask import Flask, render_template, request, jsonify
from Agent import Agent
import logging
import requests
import base64
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
agent = Agent()

def fetch_image(image_url):
    """Fetch image from URL and convert to base64 for display"""
    try:
        response = requests.get(image_url, timeout=10)
        if response.status_code == 200:
            return base64.b64encode(response.content).decode('utf-8')
        return None
    except:
        return None

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get response from the agent
        first_response = agent.first_prompt_pass(user_message)
        second_response_str = agent.second_prompt_pass(first_response)
        
        '''
        # Parse the JSON response from the agent
        try:
            second_response = json.loads(second_response_str)
            print(second_response)
        except:
            # If parsing fails, return the raw text response
            return jsonify({
                'response': second_response_str,
                'status': 'failed'
            })
        
        for item in second_response.values():
            # Assuming there's a function to fetch the image URL and return the image data
            
            print(item['Image'])
            image_data = fetch_image(item['Image'])
            # Convert the image data to a format suitable for display in the chat
            item['image_data'] = image_data

        # compose the response with the image data
        composed_response = []
        for item in second_response.values():
            composed_item = {
                'name': item['Item'],
                'category': item['Category'],
                'price': item['Price'], 
                'image_data': item['image_data']
            }
        '''
        return jsonify({
            'response': second_response_str,
            'status': 'success'
        })
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({
            'error': 'An error occurred while processing your request',
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 