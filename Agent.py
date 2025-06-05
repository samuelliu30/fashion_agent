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
    
    def get_completion_from_messages(self, messages: list) -> str:
        """
        This method takes a list of messages and returns a completion based on those messages.
        It uses the OpenAI API to generate a response.
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature = 0.0, 
                max_tokens=1000, 
            )
            completion = response.choices[0].message['content']
            logger.info(f"Completion generated from messages: {completion}")
            return completion
        except Exception as e:
            logger.error(f"Error generating completion from messages: {e}")
            return "Error generating completion"
        
    def compose_prompt(self, user_message: str) -> List[Dict[str, str]]:
        """
        This method takes a user message and composes a complete prompt with roles.
        It returns a list of message dictionaries, each with a 'role' and 'content'.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message}
        ]
        return messages
    


if __name__ == "__main__":
    # Example usage
    agent = Agent(api_key="your-api-key-here")
    result = agent.run("Hello, how are you?")
    print(result)
