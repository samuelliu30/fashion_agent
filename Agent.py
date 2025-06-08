# Fashion AI Agent - Handles conversations and product recommendations
import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import openai
import json
import requests
from datetime import datetime
import re
import random

# Dynamic model import based on configuration
from config import SEARCH_METHOD
if SEARCH_METHOD == 'semantic':
    from Fashion_model_semantic import fashion_api
    logger = logging.getLogger(__name__)
    logger.info("SEMANTIC SEARCH: Using advanced vector embeddings")
else:
    from Fashion_model import fashion_api
    logger = logging.getLogger(__name__)
    logger.info("TF-IDF SEARCH: Using traditional text vectorization")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class Agent:
    """Main fashion AI agent that handles conversations and recommendations"""
    
    def __init__(self):
        self.fashion_api = fashion_api  # Connect to product database
        self.conversation_context = []   # Store conversation history
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
            completion = response.choices[0].message['content']
            logger.info(f"Completion generated successfully")
            return completion
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            return "I apologize, but I'm having trouble processing your request right now. Please try again."
    
    def classify_user_intent(self, user_message: str) -> Dict[str, Any]:
        """
        Use AI to understand what the user wants (shopping, advice, chat, etc.)
        Replaces rigid keyword matching with intelligent understanding
        """
        intent_system_message = """
        You are an intelligent intent classifier for a fashion AI assistant. Your job is to analyze user messages and determine their intent.

        Classify user messages into these categories:

        1. **fashion_request**: User wants specific product recommendations, outfits, or shopping help
           - Examples: "I need a dress for work", "show me casual shirts", "something for a date", "outfit under $100"

        2. **fashion_advice**: User wants general fashion tips, styling advice, or fashion discussion
           - Examples: "fashion tips for summer", "how to style a blazer", "what colors go well together"

        3. **conversational**: Standard greetings, gratitude, personal questions, small talk
           - Examples: "hello", "thank you", "how are you", "what's your name"

        4. **technical_question**: User wants to know how the recommendation system works
           - Examples: "how do you find products", "explain your process", "how does your AI work"

        5. **non_fashion_redirect**: Topics completely unrelated to fashion that should be redirected
           - Examples: "what's the weather", "who is the president", "tell me about cars"

        6. **clarification_needed**: Intent is unclear and requires follow-up questions
           - Examples: very short ambiguous messages, unclear requests

        Consider the conversation history to understand context and clarifications.
        
        Respond ONLY with a JSON object:
        {
            "intent": "category_name",
            "confidence": 0.95,
            "reasoning": "Brief explanation of why this intent was chosen"
        }
        """

        # Build conversation context for better understanding
        context_messages = []
        if len(self.conversation_context) > 1:
            # Include last 5 messages for context (excluding current one)
            recent_context = self.conversation_context[-5:]
            context_str = "\n".join([f"Previous: {msg}" for msg in recent_context])
            context_messages = [
                {"role": "system", "content": intent_system_message},
                {"role": "user", "content": f"Conversation history:\n{context_str}\n\nCurrent message to classify: '{user_message}'"}
            ]
        else:
            context_messages = [
                {"role": "system", "content": intent_system_message},
                {"role": "user", "content": f"Classify this message: '{user_message}'"}
            ]

        messages = context_messages

        try:
            intent_response = self.get_completion_from_messages(messages, temperature=0.1, max_tokens=200)
            intent_data = json.loads(intent_response)
            
            # Validate AI response
            required_fields = ['intent', 'confidence', 'reasoning']
            if not all(field in intent_data for field in required_fields):
                raise ValueError("Invalid response format from intent classifier")
            
            # Add compatibility flags for existing code
            intent_data.update({
                'message_length': len(user_message.split()),
                'is_fashion_request': intent_data['intent'] == 'fashion_request',
                'is_fashion_advice': intent_data['intent'] == 'fashion_advice',
                'is_conversational': intent_data['intent'] == 'conversational',
                'is_technical_question': intent_data['intent'] == 'technical_question',
                'is_non_fashion': intent_data['intent'] == 'non_fashion_redirect',
                'is_clarification_needed': intent_data['intent'] == 'clarification_needed'
            })
            
            logger.info(f"AI Intent Classification: {intent_data['intent']} (confidence: {intent_data['confidence']}) - {intent_data['reasoning']}")
            return intent_data

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Error in AI intent classification: {e}")
            
            # Simple fallback if AI fails
            message_lower = user_message.lower().strip()
            
            if any(word in message_lower for word in ['need', 'want', 'looking for', 'show me', 'find', 'outfit', 'dress', 'shirt', 'pants', 'clothes']):
                fallback_intent = 'fashion_request'
            elif any(word in message_lower for word in ['tip', 'advice', 'how to', 'style', 'fashion', 'trend']):
                fallback_intent = 'fashion_advice'
            elif any(word in message_lower for word in ['hello', 'hi', 'thank', 'how are you', 'what', 'who']):
                fallback_intent = 'conversational'
            else:
                fallback_intent = 'conversational'
            
            return {
                'intent': fallback_intent,
                'confidence': 0.5,
                'reasoning': 'Fallback classification due to AI error',
                'message_length': len(user_message.split()),
                'is_fashion_request': fallback_intent == 'fashion_request',
                'is_fashion_advice': fallback_intent == 'fashion_advice',
                'is_conversational': fallback_intent == 'conversational',
                'is_technical_question': False,
                'is_non_fashion': False,
                'is_clarification_needed': False,
                'fallback_used': True
            }
    
    def handle_conversational_response(self, user_message: str, intent_info: Dict) -> Dict[str, Any]:
        """
        Handle friendly conversation (greetings, thanks, questions about the AI)
        """
        message_lower = user_message.lower().strip()
        
        # Different response types with friendly, engaging messages
        responses = {
            'name_identity': [
                "I'm your personal fashion stylist! You can call me your Style Assistant. I'm here to help you discover amazing outfits that make you feel confident and fabulous! What style are you in the mood for today? ✨",
                "Hi! I'm your dedicated fashion consultant - think of me as your style-savvy friend who knows all the latest trends! What kind of outfit are you dreaming of? 💫",
                "I'm your AI fashion stylist! I love helping people discover their unique style and find outfits that make them shine. What's the occasion? 👗"
            ],
            'fashion_advice': [
                "Here's a great fashion tip! 💡 Always remember the power of accessories - they can completely transform a basic outfit! A simple white t-shirt and jeans can become chic with the right statement necklace. ✨",
                "Fashion tip time! 👗 One rule I swear by: invest in quality basics. A well-fitted pair of jeans, classic white button-down, and a little black dress will be your wardrobe's foundation! 💫",
                "Here's my favorite styling tip! 🌟 Don't be afraid to mix textures and patterns, but keep one element consistent - like color palette. It creates visual interest while staying cohesive! 🎨",
                "Fashion wisdom coming your way! ✨ The fit is everything! A $20 shirt that fits perfectly will always look better than an expensive one that's too big or small. 👌",
                "Style tip alert! 💕 Create a capsule wardrobe with pieces that all work together. Choose 3-4 colors you love, then build around that. Everything mixes and matches! 🌈"
            ],
            'thank': [
                "You're very welcome! I'm so happy I could help you find the perfect style! 😊",
                "My pleasure! I hope you love the outfit recommendations. Feel free to ask if you need any styling tips!",
                "You're welcome! I'm here whenever you need fashion advice. Have a wonderful time wearing your new style! ✨"
            ],
            'greeting': [
                "Hello! I'm your personal fashion stylist! Ready to create some amazing outfits together? Tell me what you're looking for! 👗",
                "Hi there! I'm excited to help you discover your perfect style. What kind of look are you going for today?",
                "Hey! Welcome to your personal styling session. What fashion adventure shall we embark on? 🌟"
            ],
            'goodbye': [
                "Goodbye! Remember, great style is just a conversation away. Come back anytime for more fashion inspiration! 💕",
                "See you later! Keep rocking those amazing outfits, and don't hesitate to return for more styling magic!",
                "Bye! May your wardrobe always be fabulous! Feel free to visit again for your next fashion journey! ✨"
            ],
            'positive': [
                "I'm so glad you're happy with the recommendations! That's what makes being a fashion stylist so rewarding! 😊",
                "Yay! I love when we find the perfect match. You're going to look absolutely stunning! 💃",
                "That makes me so happy! Fashion should make you feel confident and amazing! ✨"
            ],
            'general': [
                "I'm doing wonderful, thank you for asking! I'm here and ready to help you create some fabulous outfits! What's your fashion mood today?",
                "I'm great! I've been helping people discover their perfect style all day, and I'd love to help you too! What are you looking to wear?",
                "I'm fantastic! There's nothing I love more than talking fashion and style. What can I help you find today? 👗"
            ],
            'non_fashion_redirect': [
                "That's an interesting topic! However, I'm your dedicated fashion stylist, so I'm most helpful when we're talking about style and outfits! 👗 What kind of look are you going for?",
                "I appreciate the question, but my expertise is all about fashion and style! I'd love to help you put together an amazing outfit instead. What occasion are you dressing for? ✨",
                "While that's a great question, I'm your personal fashion consultant! Let's focus on making you look fabulous! 💫 What's your next fashion challenge?",
            ]
        }
        
        # Determine response category based on message content
        if intent_info.get('intent') == 'non_fashion_redirect':
            category = 'non_fashion_redirect'
        elif re.search(r'\b(what.*your name|who are you|what are you|introduce yourself)\b', message_lower):
            category = 'name_identity'
        elif intent_info.get('is_fashion_advice', False):
            category = 'fashion_advice'
        elif re.search(r'\b(thank|thx)\b', message_lower):
            category = 'thank'
        elif re.search(r'\b(hello|hi|hey)\b', message_lower):
            category = 'greeting'
        elif re.search(r'\b(bye|goodbye)\b', message_lower):
            category = 'goodbye'
        elif re.search(r'\b(good|great|awesome|nice|perfect|love|amazing)\b', message_lower):
            category = 'positive'
        elif re.search(r'\b(how are you|how\'s it going)\b', message_lower):
            category = 'general'
        else:
            category = 'general'
        
        # Pick a random response from the appropriate category
        response = random.choice(responses.get(category, responses['general']))
        
        return {
            'status': 'success',
            'message_type': 'conversational',
            'response': response,
            'category': category,
            'user_query': user_message
        }

    def handle_explanation_request(self, user_message: str) -> Dict[str, Any]:
        """
        Explain how the fashion recommendation system works
        """
        explanation = """
        I'd love to explain how I work! ✨

        🧠 **My Brain**: I use advanced AI technology called RAG (Retrieval-Augmented Generation) combined with TF-IDF vectorization to understand your style needs and find perfect matches.

        🔍 **Smart Search**: When you tell me what you're looking for, I:
        1. Analyze your request using AI to understand your style, occasion, and budget
        2. Search through thousands of Zara products using semantic similarity 
        3. Match your needs with the perfect items based on descriptions, categories, and style attributes

        💡 **Two-Step Process**:
        - **First**: I analyze your style preferences and create a comprehensive style profile
        - **Second**: I use that analysis to find specific products that match your needs

        🎯 **Smart Features**:
        - Budget-aware recommendations (I respect your spending limits!)
        - Occasion-appropriate suggestions (work, dates, casual, etc.)
        - Style matching based on your personality and preferences
        - Real product availability from Zara's current inventory

        🔗 **Clickable Shopping**: Every product I recommend is clickable - just click any item to shop directly on Zara's website!

        The magic happens through machine learning that understands fashion language and matches it with real products. Think of me as your AI stylist with access to thousands of outfits! 🌟

        What would you like to shop for today?
        """
        
        return {
            'status': 'success',
            'message_type': 'explanation',
            'response': explanation,
            'user_query': user_message
        }

    def analyze_user_request(self, user_message: str) -> Dict[str, Any]:
        """
        Extract key information from user's fashion request (budget, style, occasion)
        """
        try:
            # Extract budget information
            budget_range = self.fashion_api.extract_budget_from_text(user_message)
            
            # Find matching styles using AI similarity
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
                'raw_message': user_message,
                'error': str(e)
            }

    def first_prompt_pass(self, user_message: str) -> str:
        """
        First AI pass: Analyze user's style preferences and create detailed style profile
        """
        delimiter = "####"
        
        # Build context-aware prompt
        context_info = ""
        if len(self.conversation_context) > 1:
            recent_context = self.conversation_context[-5:]
            context_info = f"\n\nConversation History (for context):\n" + "\n".join([f"- {msg}" for msg in recent_context])
        
        system_message = f"""
        You are an expert fashion stylist analyzing a customer's style request. Your job is to create a comprehensive style analysis.

        Analyze the user's request and provide:
        1. **Style Classification**: What style category best fits (casual, business, formal, trendy, etc.)
        2. **Occasion Analysis**: What occasion/setting is this for
        3. **Key Requirements**: Specific items mentioned or implied
        4. **Style Personality**: What vibe/personality should the outfit convey
        5. **Categories**: List specific clothing categories to search for (e.g., ["shirts", "pants", "dresses", "jackets", "shoes"])
        6. **Gender Considerations**: If specified or clarified, ensure recommendations match the correct gender

        Format your response as a detailed analysis that will help in product selection.
        Be specific about the style direction and clothing categories needed.
        
        **IMPORTANT**: For outfit requests, always end with:
        Categories: ["category1", "category2", "category3", ...]
        
        Use categories like: shirts, tops, pants, jeans, dresses, skirts, jackets, blazers, shoes, sneakers, boots
        For complete outfits, provide 3-5 diverse categories to build a cohesive look.
        Use conversation history to understand clarifications or additional context.

        Current Request: {delimiter}{user_message}{delimiter}{context_info}
        """
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"{delimiter}Analyze this style request{delimiter}"}
        ]
        
        return self.get_completion_from_messages(messages, temperature=0.2)

    def second_prompt_pass(self, style_analysis: str, user_query: str) -> str:
        """
        Second AI pass: Find specific products and create personalized recommendations
        """
        delimiter = "####"
        
        # Get user preferences
        analysis_result = self.analyze_user_request(user_query)
        budget_range = analysis_result.get('budget')
        
        # Detect what types of products the user wants
        product_types = self._detect_product_types(user_query)
        logger.info(f"User request analyzed: budget={budget_range}, product_types={product_types}")
        
        # Extract categories from style analysis
        categories = self._extract_categories_from_analysis(style_analysis)
        
        # Search for products using extracted categories or fallback to text search
        try:
            if categories:
                products_df = self.fashion_api.search_products_by_category(
                    categories, 
                    max_per_category=3, 
                    budget_range=budget_range
                )
            else:
                # For outfit requests, search with outfit-building keywords
                if 'outfit' in user_query.lower() or 'clothing' in product_types:
                    # Build a complete outfit by searching multiple categories
                    outfit_categories = ['shirts', 'tops', 'pants', 'jeans', 'dresses', 'jackets', 'blazers']
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
            
            # Filter products by detected types
            if products_df is not None and hasattr(products_df, 'empty') and not products_df.empty:
                products_df = self._filter_products_by_type(products_df, product_types)
                products_data = self._format_products_for_response(products_df)
            else:
                products_data = []
        except Exception as e:
            logger.error(f"Error searching for products: {e}")
            products_data = []
        
        # Handle case where no products are found
        if not products_data:
            return json.dumps({
                "message": "I'm so sorry, but I couldn't find products matching your exact criteria in our current inventory. This might be because of specific budget constraints or very particular style requirements.",
                "suggestions": "Would you like to try adjusting your budget slightly or exploring a similar style? I'd love to help you find something amazing!",
                "products": []
            })
        
        # Create AI prompt for generating personalized recommendations
        product_types_str = ", ".join(product_types) if product_types else "clothing"
        
        # Add conversation context for better understanding
        context_info = ""
        if len(self.conversation_context) > 1:
            recent_context = self.conversation_context[-5:]
            context_info = f"\n\nConversation Context:\n" + "\n".join([f"- {msg}" for msg in recent_context])
        
        system_message = f"""
        You are an enthusiastic fashion stylist presenting curated recommendations with warmth and personality.
        
        The user asked for: {product_types_str}
        Based on the style analysis: {style_analysis}{context_info}
        
        I have found {len(products_data)} products from our Zara inventory that match their request.
        
        Format your response as a JSON object with this exact structure:
        {{
            "style_summary": "Brief, warm summary of the recommended style/products with personal enthusiasm",
            "total_outfit_price": "Total price range for all recommended items",
            "products": [
                {{
                    "name": "Product name",
                    "category": "Product category (clothing/perfume/etc)", 
                    "price": "Price with currency",
                    "description": "Brief style description",
                    "image_url": "Product image URL",
                    "url": "Product URL for Zara website",
                    "why_chosen": "Personal, warm explanation of why this item was chosen and how it fits their request"
                }}
            ],
            "styling_tips": "3-4 warm, personal tips for styling/using these items - be encouraging and enthusiastic"
        }}
        
        Products found in inventory:
        {json.dumps(products_data, indent=2)}
        
        Create an enthusiastic, warm, and personal response that explains why each item was chosen. If there are both clothing and perfume items, explain how they complement each other. Sound like a friend who's genuinely excited about helping them look and feel amazing!
        """
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"{delimiter}Create the outfit recommendation{delimiter}"}
        ]
        
        return self.get_completion_from_messages(messages, temperature=0.3)
    
    def _extract_categories_from_analysis(self, analysis_text: str) -> List[str]:
        """Extract clothing categories from AI style analysis"""
        import re
        
        # Look for Categories: [list] pattern in the analysis
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
        perfume_keywords = ['perfume', 'fragrance', 'scent', 'cologne', 'eau de parfum', 'edp', 'eau de toilette', 'edt']
        if any(keyword in query_lower for keyword in perfume_keywords):
            detected_types.append('perfume')
        
        # Clothing indicators
        clothing_keywords = ['outfit', 'shirt', 'pants', 'dress', 'jacket', 'top', 'bottom', 'clothes', 'clothing', 'wear', 'attire']
        if any(keyword in query_lower for keyword in clothing_keywords):
            detected_types.append('clothing')
        
        # If no specific type detected, assume clothing (most common request)
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
                    products_df['name'].str.contains(r'EDP|EAU DE|PARFUM|PERFUME|FRAGRANCE|ML|FL\. OZ', case=False, na=False, regex=True) |
                    products_df['description'].str.contains(r'perfume|fragrance|eau de|scent|notes of', case=False, na=False, regex=True)
                )
                perfume_products = products_df[perfume_mask]
                if not perfume_products.empty:
                    filtered_products.append(perfume_products.head(3))
            
            elif requested_type == 'clothing':
                # Filter for clothing items (exclude perfumes)
                clothing_mask = ~(
                    products_df['name'].str.contains(r'EDP|EAU DE|PARFUM|PERFUME|FRAGRANCE|ML|FL\. OZ', case=False, na=False, regex=True) |
                    products_df['description'].str.contains(r'perfume|fragrance|eau de|scent|notes of', case=False, na=False, regex=True)
                )
                clothing_products = products_df[clothing_mask]
                if not clothing_products.empty:
                    filtered_products.append(clothing_products.head(4))
        
        # Combine filtered products
        if filtered_products:
            import pandas as pd
            return pd.concat(filtered_products, ignore_index=True)
        else:
            return products_df.head(0)  # Return empty dataframe
    
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
                description = str(description)  # Ensure it's a string
            
            # Truncate description if too long
            if len(description) > 200:
                description = description[:200] + "..."
            
            product_dict = {
                "name": str(product.get('name', '')),
                "category": str(product.get('terms', 'general')),
                "price": f"${float(product.get('price', 0)):.2f}",
                "description": description,
                "image_url": str(product.get('primary_image', '')),
                "url": str(product.get('url', '')),  # Include URL for clickable links
                "section": str(product.get('section', '')),
                "similarity_score": float(product.get('similarity_score', 0))
            }
            products_list.append(product_dict)
        
        return products_list
    
    def process_user_request(self, user_message: str) -> Dict[str, Any]:
        """
        Main processing pipeline: understand intent, analyze request, generate recommendations
        """
        try:
            # Store conversation for context
            self.conversation_context.append(user_message)
            
            # Use AI to understand what user wants
            intent_info = self.classify_user_intent(user_message)
            logger.info(f"User intent classified as: {intent_info['intent']} (confidence: {intent_info.get('confidence', 0)})")
            
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
                # Step 1: Analyze style preferences
                style_analysis = self.first_prompt_pass(user_message)
                
                # Step 2: Find products and create recommendations
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
                'message': 'Oh no! I encountered a little hiccup while processing your request. Could you try rephrasing that for me? I promise I\'ll do my best to help! 😊',
                'error': str(e)
            }

    def handle_clarification_request(self, user_message: str, intent_info: Dict) -> Dict[str, Any]:
        """
        Handle cases where user intent is unclear and we need to ask follow-up questions
        """
        clarification_questions = [
            "I'd love to help you with that! Could you tell me a bit more about what you're looking for? Are you shopping for a specific item or occasion? 🌟",
            "I want to make sure I give you the perfect recommendations! Could you share more details about what kind of style or items you have in mind? ✨",
            "Let me help you find something amazing! What specifically are you looking for - maybe an outfit for a particular occasion or a certain type of clothing? 💫"
        ]
        
        # Use AI-suggested clarification if available, otherwise use default
        if intent_info.get('potential_clarification'):
            response = intent_info['potential_clarification']
        else:
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
