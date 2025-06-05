import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import openai
import json
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class Agent:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = api_key

    def run(self, prompt):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()

    def chat(self, messages):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return response.choices[0].message['content']

    def load_inventory_database(self, inventory_path: str) -> Dict[str, Any]:
        """Load inventory database from the specified path."""
        try:
            with open(inventory_path, 'r') as f:
                inventory_data = json.load(f)
            logger.info(f"Inventory database loaded from {inventory_path}")
            return inventory_data
        except Exception as e:
            logger.error(f"Error loading inventory database: {e}")
            return {}

    def load_style_database(self, style_path: str) -> Dict[str, Any]:
        """Load style database from the specified path."""
        try:
            with open(style_path, 'r') as f:
                style_data = json.load(f)
            logger.info(f"Style database loaded from {style_path}")
            return style_data
        except Exception as e:
            logger.error(f"Error loading style database: {e}")
            return {}

if __name__ == "__main__":
    # Example usage
    agent = Agent(api_key="your-api-key-here")
    result = agent.run("Hello, how are you?")
    print(result)
