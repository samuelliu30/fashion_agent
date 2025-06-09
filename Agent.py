# Fashion AI Agent - Handles conversations and product recommendations
import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import openai
import json
import re
import random

# Dynamic model import based on configuration
from config import SEARCH_METHOD
if SEARCH_METHOD == 'semantic':
    from Fashion_model_semantic import fashion_api
else:
    from Fashion_model import fashion_api

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info(f"Fashion Agent initialized with {SEARCH_METHOD.upper()} search")

# Load OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class Agent:
    """Main fashion AI agent that handles conversations and recommendations"""
    
    def __init__(self):
        self.fashion_api = fashion_api
        self.conversation_context = []
        logger.info("Fashion Agent initialized with RAG capabilities")
    
    def get_completion_from_messages(self, messages: list, temperature=0.0, max_tokens=1500) -> str:
        """Generate AI response using OpenAI GPT"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=temperature, 
                max_tokens=max_tokens, 
            )
            return response.choices[0].message['content']
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            return "I apologize, but I'm having trouble processing your request right now. Please try again."
    
    def classify_user_intent(self, user_message: str) -> Dict[str, Any]:
        """Use AI to understand what the user wants (shopping, advice, chat, etc.)"""
        intent_system_message = """
        You are an intelligent intent classifier for a fashion AI assistant. 

        Classify user messages into these categories:
        1. **fashion_request**: User wants specific product recommendations or shopping help
        2. **fashion_advice**: User wants general fashion tips or styling advice  
        3. **conversational**: Greetings, gratitude, personal questions, small talk
        4. **technical_question**: User wants to know how the system works
        5. **non_fashion_redirect**: Topics unrelated to fashion
        6. **clarification_needed**: Intent is unclear and requires follow-up

        Respond ONLY with a JSON object:
        {
            "intent": "category_name",
            "confidence": 0.95,
            "reasoning": "Brief explanation"
        }
        """

        # Build conversation context
        context_messages = [{"role": "system", "content": intent_system_message}]
        
        if len(self.conversation_context) > 1:
            recent_context = self.conversation_context[-5:]
            context_str = "\n".join([f"Previous: {msg}" for msg in recent_context])
            context_messages.append({
                "role": "user", 
                "content": f"Conversation history:\n{context_str}\n\nCurrent message: '{user_message}'"
            })
        else:
            context_messages.append({"role": "user", "content": f"Classify: '{user_message}'"})

        try:
            intent_response = self.get_completion_from_messages(context_messages, temperature=0.1, max_tokens=200)
            intent_data = json.loads(intent_response)
            
            # Validate response
            if not all(field in intent_data for field in ['intent', 'confidence', 'reasoning']):
                raise ValueError("Invalid AI response format")
            
            logger.info(f"Intent: {intent_data['intent']} (confidence: {intent_data['confidence']})")
            return intent_data

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error in AI intent classification: {e}")
            
            # Simple fallback
            message_lower = user_message.lower().strip()
            
            if any(word in message_lower for word in ['need', 'want', 'looking for', 'show me', 'find', 'outfit']):
                fallback_intent = 'fashion_request'
            elif any(word in message_lower for word in ['tip', 'advice', 'how to', 'style']):
                fallback_intent = 'fashion_advice'
            else:
                fallback_intent = 'conversational'
            
            return {
                'intent': fallback_intent,
                'confidence': 0.5,
                'reasoning': 'Fallback classification due to AI error'
            }
    
    def handle_conversational_response(self, user_message: str, intent_info: Dict) -> Dict[str, Any]:
        """Handle friendly conversation (greetings, thanks, general questions)"""
        message_lower = user_message.lower().strip()
        
        # Simplified response categories
        responses = {
            'greeting': [
                "Hello! I'm your personal fashion stylist! Ready to create some amazing outfits together? Tell me what you're looking for! 👗",
                "Hi there! I'm excited to help you discover your perfect style. What kind of look are you going for today?",
            ],
            'thank': [
                "You're very welcome! I'm so happy I could help you find the perfect style! 😊",
                "My pleasure! I hope you love the recommendations. Feel free to ask if you need more styling tips!",
            ],
            'fashion_advice': [
                "Here's a great fashion tip! 💡 Always remember the power of accessories - they can transform any basic outfit!",
                "Fashion tip: invest in quality basics! Well-fitted jeans, a classic white shirt, and a little black dress are wardrobe foundations! ✨",
            ],
            'non_fashion': [
                "That's interesting! However, I'm your fashion stylist, so I'm most helpful with style and outfits! 👗 What look are you going for?",
                "I appreciate the question, but my expertise is fashion! Let's focus on making you look fabulous! What's your next style challenge? ✨",
            ],
            'general': [
                "I'm doing wonderful! I'm here and ready to help you create some fabulous outfits! What's your fashion mood today?",
                "I'm fantastic! There's nothing I love more than talking fashion. What can I help you find today? 👗",
            ]
        }
        
        # Determine response category
        if intent_info.get('intent') == 'non_fashion_redirect':
            category = 'non_fashion'
        elif intent_info.get('intent') == 'fashion_advice':
            category = 'fashion_advice'
        elif re.search(r'\b(thank|thx)\b', message_lower):
            category = 'thank'
        elif re.search(r'\b(hello|hi|hey)\b', message_lower):
            category = 'greeting'
        else:
            category = 'general'
        
        response = random.choice(responses[category])
        
        return {
            'status': 'success',
            'message_type': 'conversational',
            'response': response,
            'user_query': user_message
        }

    def handle_explanation_request(self, user_message: str) -> Dict[str, Any]:
        """Explain how the fashion recommendation system works"""
        explanation = """
        I'd love to explain how I work! ✨

        🧠 **My Brain**: I use advanced AI (GPT-3.5) with RAG (Retrieval-Augmented Generation) to understand your style needs.

        🔍 **Smart Search**: When you tell me what you're looking for, I:
        1. Analyze your request using AI to understand style, occasion, and budget
        2. Search through thousands of Zara products using semantic similarity 
        3. Match your needs with perfect items based on descriptions and categories

        💡 **Two-Step Process**:
        - **First**: Analyze your style preferences and create a style profile
        - **Second**: Use that analysis to find specific products that match

        🎯 **Smart Features**:
        - Budget-aware recommendations
        - Occasion-appropriate suggestions  
        - Style matching based on your preferences
        - Real product availability from Zara's inventory

        🔗 **Clickable Shopping**: Every recommendation is clickable - shop directly on Zara's website!

        Think of me as your AI stylist with access to thousands of outfits! 🌟
        What would you like to shop for today?
        """
        
        return {
            'status': 'success',
            'message_type': 'explanation',
            'response': explanation,
            'user_query': user_message
        }

    def analyze_user_request(self, user_message: str) -> Dict[str, Any]:
        """Extract key information from user's fashion request (budget, style, occasion)"""
        try:
            budget_range = self.fashion_api.extract_budget_from_text(user_message)
            matching_styles = self.fashion_api.find_matching_styles(user_message, max_results=3)
            
            return {
                'budget': budget_range,
                'styles_found': len(matching_styles),
                'matching_styles': matching_styles.to_dict('records') if not matching_styles.empty else [],
                'raw_message': user_message
            }
        except Exception as e:
            logger.error(f"Error analyzing user request: {e}")
            return {
                'budget': None,
                'styles_found': 0,
                'matching_styles': [],
                'raw_message': user_message
            }

    def first_prompt_pass(self, user_message: str) -> str:
        """First AI pass: Analyze user's style preferences and create detailed style profile"""
        delimiter = "####"
        
        # Build context-aware prompt
        context_info = ""
        if len(self.conversation_context) > 1:
            recent_context = self.conversation_context[-5:]
            context_info = f"\n\nConversation History:\n" + "\n".join([f"- {msg}" for msg in recent_context])
        
        system_message = f"""
        You are an expert fashion stylist analyzing a customer's style request.

        Analyze the user's request and provide:
        1. **Style Classification**: What style category best fits (casual, business, formal, trendy, etc.)
        2. **Occasion Analysis**: What occasion/setting is this for
        3. **Key Requirements**: Specific items mentioned or implied
        4. **Style Personality**: What vibe should the outfit convey
        5. **Categories**: List specific clothing categories to search for

        **IMPORTANT**: Always end with:
        Categories: ["category1", "category2", "category3"]
        
        Use categories like: shirts, tops, pants, jeans, dresses, skirts, jackets, blazers, shoes, sneakers, boots
        For complete outfits, provide 3-5 diverse categories.

        Current Request: {delimiter}{user_message}{delimiter}{context_info}
        """
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"{delimiter}Analyze this style request{delimiter}"}
        ]
        
        return self.get_completion_from_messages(messages, temperature=0.2)

    def second_prompt_pass(self, style_analysis: str, user_query: str) -> str:
        """Second AI pass: Find specific products and create personalized recommendations"""
        delimiter = "####"
        
        # Get user preferences
        analysis_result = self.analyze_user_request(user_query)
        budget_range = analysis_result.get('budget')
        
        # Detect product types and extract categories
        product_types = self._detect_product_types(user_query)
        categories = self._extract_categories_from_analysis(style_analysis)
        
        logger.info(f"Search parameters: budget={budget_range}, product_types={product_types}, categories={categories}")
        
        # Search for products
        try:
            if categories:
                products_df = self.fashion_api.search_products_by_category(
                    categories, 
                    max_per_category=3, 
                    budget_range=budget_range
                )
            else:
                # Fallback to text search
                if 'outfit' in user_query.lower():
                    outfit_categories = ['shirts', 'tops', 'pants', 'jeans', 'dresses', 'jackets']
                    products_df = self.fashion_api.search_products_by_category(
                        outfit_categories, 
                        max_per_category=2, 
                        budget_range=budget_range
                    )
                else:
                    products_df = self.fashion_api.search_products_by_text(
                        user_query, 
                        max_results=6, 
                        budget_range=budget_range
                    )
            
            # Filter and format products
            if products_df is not None and not products_df.empty:
                products_df = self._filter_products_by_type(products_df, product_types)
                products_data = self._format_products_for_response(products_df)
            else:
                products_data = []
                
        except Exception as e:
            logger.error(f"Error searching for products: {e}")
            products_data = []
        
        # Handle no products found
        if not products_data:
            return json.dumps({
                "message": "I couldn't find products matching your exact criteria in our current inventory. This might be due to budget constraints or specific style requirements.",
                "suggestions": "Would you like to try adjusting your budget or exploring a similar style? I'd love to help you find something amazing!",
                "products": []
            })
        
        # Build context for AI response
        product_types_str = ", ".join(product_types) if product_types else "clothing"
        context_info = ""
        if len(self.conversation_context) > 1:
            recent_context = self.conversation_context[-5:]
            context_info = f"\n\nConversation Context:\n" + "\n".join([f"- {msg}" for msg in recent_context])
        
        system_message = f"""
        You are an enthusiastic fashion stylist presenting curated recommendations.
        
        The user asked for: {product_types_str}
        Based on analysis: {style_analysis}{context_info}
        
        Found {len(products_data)} products from Zara inventory that match their request.
        
        Format response as JSON:
        {{
            "style_summary": "Brief, warm summary of recommended style with enthusiasm",
            "total_outfit_price": "Total price range for all items",
            "products": [
                {{
                    "name": "Product name",
                    "category": "Product category", 
                    "price": "Price with currency",
                    "description": "Brief style description",
                    "image_url": "Product image URL",
                    "url": "Product URL for Zara website",
                    "why_chosen": "Warm explanation of why this item was chosen"
                }}
            ],
            "styling_tips": "3-4 personal tips for styling these items"
        }}
        
        Products found:
        {json.dumps(products_data, indent=2)}
        
        Create an enthusiastic, warm response explaining why each item was chosen!
        """
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"{delimiter}Create the outfit recommendation{delimiter}"}
        ]
        
        return self.get_completion_from_messages(messages, temperature=0.3)
    
    def _extract_categories_from_analysis(self, analysis_text: str) -> List[str]:
        """Extract clothing categories from AI style analysis"""
        pattern = r"Categories:\s*\[(.*?)\]"
        match = re.search(pattern, analysis_text)
        
        if match:
            categories_str = match.group(1)
            categories = [cat.strip().strip("'\"") for cat in categories_str.split(',')]
            return [cat for cat in categories if cat]
        
        return []
    
    def _detect_product_types(self, user_query: str) -> List[str]:
        """Detect what types of products the user is asking for"""
        query_lower = user_query.lower()
        detected_types = []
        
        # Perfume/Fragrance indicators
        if any(keyword in query_lower for keyword in ['perfume', 'fragrance', 'scent', 'cologne']):
            detected_types.append('perfume')
        
        # Clothing indicators (default)
        if any(keyword in query_lower for keyword in ['outfit', 'shirt', 'pants', 'dress', 'clothes']):
            detected_types.append('clothing')
        
        # Default to clothing if nothing detected
        if not detected_types:
            detected_types.append('clothing')
        
        return detected_types
    
    def _filter_products_by_type(self, products_df, requested_types: List[str]):
        """Filter products based on requested product types"""
        if not requested_types:
            return products_df
        
        filtered_products = []
        
        for requested_type in requested_types:
            if requested_type == 'perfume':
                # Filter for perfumes/fragrances
                perfume_mask = (
                    products_df['name'].str.contains(r'EDP|EAU DE|PARFUM|PERFUME|FRAGRANCE', case=False, na=False, regex=True) |
                    products_df['description'].str.contains(r'perfume|fragrance|eau de|scent', case=False, na=False, regex=True)
                )
                perfume_products = products_df[perfume_mask]
                if not perfume_products.empty:
                    filtered_products.append(perfume_products.head(3))
            
            elif requested_type == 'clothing':
                # Filter for clothing items (exclude perfumes)
                clothing_mask = ~(
                    products_df['name'].str.contains(r'EDP|EAU DE|PARFUM|PERFUME|FRAGRANCE', case=False, na=False, regex=True) |
                    products_df['description'].str.contains(r'perfume|fragrance|eau de|scent', case=False, na=False, regex=True)
                )
                clothing_products = products_df[clothing_mask]
                if not clothing_products.empty:
                    filtered_products.append(clothing_products.head(4))
        
        # Combine filtered products
        if filtered_products:
            import pandas as pd
            return pd.concat(filtered_products, ignore_index=True)
        else:
            return products_df.head(0)
    
    def _format_products_for_response(self, products_df) -> List[Dict]:
        """Convert product data to format needed by AI recommendation system"""
        import pandas as pd
        products_list = []
        
        for _, product in products_df.iterrows():
            # Safely handle description field
            description = product.get('description', '')
            if pd.isna(description) or description is None:
                description = ''
            else:
                description = str(description)
            
            # Truncate description if too long
            if len(description) > 200:
                description = description[:200] + "..."
            
            product_dict = {
                "name": str(product.get('name', '')),
                "category": str(product.get('terms', 'general')),
                "price": f"${float(product.get('price', 0)):.2f}",
                "description": description,
                "image_url": str(product.get('primary_image', '')),
                "url": str(product.get('url', '')),
                "section": str(product.get('section', '')),
                "similarity_score": float(product.get('similarity_score', 0))
            }
            products_list.append(product_dict)
        
        return products_list
    
    def process_user_request(self, user_message: str) -> Dict[str, Any]:
        """Main processing pipeline: understand intent, analyze request, generate recommendations"""
        try:
            # Store conversation for context
            self.conversation_context.append(user_message)
            
            # Use AI to understand what user wants
            intent_info = self.classify_user_intent(user_message)
            logger.info(f"User intent: {intent_info['intent']} (confidence: {intent_info.get('confidence', 0)})")
            
            # Route to appropriate handler based on intent
            if intent_info['intent'] == 'conversational':
                return self.handle_conversational_response(user_message, intent_info)
            
            elif intent_info['intent'] == 'technical_question':
                return self.handle_explanation_request(user_message)
            
            elif intent_info['intent'] == 'non_fashion_redirect':
                return self.handle_conversational_response(user_message, intent_info)
            
            elif intent_info['intent'] == 'fashion_advice':
                return self.handle_conversational_response(user_message, intent_info)
            
            elif intent_info['intent'] == 'clarification_needed':
                return self.handle_clarification_request(user_message, intent_info)
            
            elif intent_info['intent'] == 'fashion_request':
                # Two-step AI process for fashion recommendations
                style_analysis = self.first_prompt_pass(user_message)
                product_recommendations = self.second_prompt_pass(style_analysis, user_message)
                
                # Try to parse as structured JSON response
                try:
                    recommendations_json = json.loads(product_recommendations)
                    return {
                        'status': 'success',
                        'style_analysis': style_analysis,
                        'recommendations': recommendations_json,
                        'raw_response': product_recommendations,
                        'intent_confidence': intent_info.get('confidence', 0)
                    }
                except json.JSONDecodeError:
                    return {
                        'status': 'partial_success',
                        'style_analysis': style_analysis,
                        'recommendations': None,
                        'raw_response': product_recommendations,
                        'intent_confidence': intent_info.get('confidence', 0)
                    }
            
            else:
                # Fallback to conversational response
                logger.warning(f"Unknown intent '{intent_info['intent']}', using conversational fallback")
                return self.handle_conversational_response(user_message, intent_info)
                
        except Exception as e:
            logger.error(f"Error in process_user_request: {e}")
            return {
                'status': 'error',
                'message': 'Oh no! I encountered a little hiccup. Could you try rephrasing that for me? 😊',
                'error': str(e)
            }

    def handle_clarification_request(self, user_message: str, intent_info: Dict) -> Dict[str, Any]:
        """Handle cases where user intent is unclear"""
        clarification_questions = [
            "I'd love to help you! Could you tell me more about what you're looking for? Are you shopping for a specific item or occasion? 🌟",
            "I want to give you perfect recommendations! Could you share more details about the style or items you have in mind? ✨",
            "Let me help you find something amazing! What specifically are you looking for - maybe an outfit for a particular occasion? 💫"
        ]
        
        response = random.choice(clarification_questions)
        
        return {
            'status': 'success',
            'message_type': 'clarification',
            'response': response,
            'user_query': user_message,
            'original_intent_confidence': intent_info.get('confidence', 0)
        }

# Example usage for testing
if __name__ == "__main__":
    agent = Agent()
    test_query = "I want a casual outfit for a date night under $200"
    result = agent.process_user_request(test_query)
    print("Style Analysis:", result.get('style_analysis'))
    print("Recommendations:", json.dumps(result.get('recommendations'), indent=2))
