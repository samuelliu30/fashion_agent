import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import openai
import json
import requests
from datetime import datetime
from Fashion_model import fashion_api
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

class Agent:
    def __init__(self):
        pass
    
    def get_completion_from_messages(self, messages: list) -> str:
        """
        This method takes a list of messages and returns a completion based on those messages.
        It uses the OpenAI API to generate a response.
        """
        try:
            #full_prompt = self.compose_prompt(messages)
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
    
    # First prompt pass
    # We tell the agent what the user wants and the agent will match the style with the categories
    def first_prompt_pass(self, user_message: str) -> str:
        """
        This method takes a user message and composes a complete prompt with roles.
        It returns a list of message dictionaries, each with a 'role' and 'content'.
        """
        delimiter = "####"
        
        # Get available categories from fashion API
        try:
            available_categories = fashion_api.get_categories()
        except:
            available_categories = ['t-shirts', 'jeans', 'sneakers', 'suit', 'dress shirt', 'tie', 'dress shoes', 'dresses', 'skirts', 'blouses', 'jackets', 'coats', 'sweaters', 'pants', 'shorts', 'activewear', 'accessories']
        
        system_message = f"""
        You are a fashion stylist. \
        The customer service query will be delimited with \
        {delimiter} characters.
        Classify the customer service query into a fashion style \
        and describe the style with a brief description. \
        Provide a list of clothing categories that complete the outfit. \
        
        For example, if the customer service query is "I want a casual outfit for a day out", \
        the output should be:
        Fashion style: Casual
        Description: A relaxed and comfortable outfit perfect for everyday activities
        Categories: ['t-shirts', 'jeans', 'sneakers']
        
        If the customer service query is "I want a formal outfit for a job interview", \
        the output should be:
        Fashion style: Formal
        Description: A professional and polished outfit suitable for business settings
        Categories: ['suit', 'dress shirt', 'tie', 'dress shoes']
        
        Available clothing categories: {available_categories}
        
        Please respond in a friendly, enthusiastic manner as a personal fashion consultant.
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"{delimiter}{user_message}{delimiter}"}
        ]
        return self.get_completion_from_messages(messages)

    # The second prompt pass
    # We take the categories and the description and we return a list of items that match the style
    def second_prompt_pass(self, message: str) -> List[str]:
        """
        This method takes a list of categories and a description and returns a list of items that match the style
        """
        system_message = f"""
        You are a fashion stylist working at Zara.
        The catalog data is:
        {fashion_api.get_catalog_data()}
        You are given a list of categories and a description \
        and you need to pick two items from each of the categories in our catalog to match the description. \
        The items should be in a json format with the following format: \
        Item: <item_name> 
        Category: <category> 
        Price: <price> 
        Image: <image_url> \
        The image_url is the first string in the image column of the of the item. \
        """
        delimiter = "####"
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"{delimiter}{message}{delimiter}"}
        ]
        return self.get_completion_from_messages(messages)

if __name__ == "__main__":
    # Example usage
    agent = Agent()
    #result = agent.get_completion_from_messages("Hello, how are you?")

    result = agent.first_prompt_pass("I want a casual outfit for a dating night")
    print(result)
