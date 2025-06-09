# Fashion Data API - SEMANTIC VERSION with Vector Embeddings
import pandas as pd
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import logging
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FashionDataAPI:
    """SEMANTIC VERSION: Advanced AI-powered fashion search using vector embeddings"""
    
    def __init__(self):
        # Load product and style data from CSV files
        csv_path = os.path.join('Data', 'store_zara.csv')
        style_csv_path = os.path.join('Data', 'Common_Outfit_Styles.csv')
        
        # Load Zara product catalog
        logger.info("Loading fashion catalog data...")
        self.catalog_data = pd.read_csv(csv_path)
        self._prepare_catalog_data()
        
        # Load outfit style definitions
        self.style_data = pd.read_csv(style_csv_path)
        
        # Set up semantic search capabilities
        self._initialize_semantic_search()
        
        logger.info(f"Loaded {len(self.catalog_data)} products and {len(self.style_data)} styles")
    
    def _prepare_catalog_data(self):
        """Clean and prepare product data for semantic search"""
        # Remove unnecessary columns
        columns_to_drop = ["brand", "sku", "currency", "scraped_at", "image_downloads", "error"]
        self.catalog_data.drop(columns=[col for col in columns_to_drop if col in self.catalog_data.columns], inplace=True)
        
        # Fix price format
        self.catalog_data['price'] = pd.to_numeric(self.catalog_data['price'], errors='coerce')
        
        # Create rich searchable text for semantic understanding
        self.catalog_data['search_text'] = (
            self.catalog_data['name'].fillna('') + '. ' +
            self.catalog_data['description'].fillna('') + '. ' +
            self.catalog_data['terms'].fillna('') + '. ' +
            self.catalog_data['section'].fillna('') + '.'
        ).str.lower()
        
        # Get first image URL for display
        self.catalog_data['primary_image'] = self.catalog_data['images'].apply(self._extract_first_image)
        
        # Remove products without essential info
        self.catalog_data = self.catalog_data.dropna(subset=['name', 'price'])
        
        logger.info(f"Catalog data prepared: {len(self.catalog_data)} valid products")
    
    def _extract_first_image(self, images_str):
        """Get the first image URL from product images"""
        try:
            if pd.isna(images_str):
                return None
            images_list = ast.literal_eval(images_str)
            return images_list[0] if images_list else None
        except:
            return None
    
    def _initialize_semantic_search(self):
        """Set up semantic search using sentence transformers"""
        logger.info("Initializing semantic search...")
        
        # Load pre-trained semantic model
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Check if we have cached embeddings
        product_embeddings_path = 'Data/product_embeddings.pkl'
        style_embeddings_path = 'Data/style_embeddings.pkl'
        
        if os.path.exists(product_embeddings_path) and os.path.exists(style_embeddings_path):
            logger.info("Loading cached embeddings...")
            with open(product_embeddings_path, 'rb') as f:
                self.product_embeddings = pickle.load(f)
            with open(style_embeddings_path, 'rb') as f:
                self.style_embeddings = pickle.load(f)
        else:
            logger.info("Creating new embeddings...")
            
            # Create semantic embeddings for all products
            product_texts = self.catalog_data['search_text'].fillna('').tolist()
            self.product_embeddings = self.semantic_model.encode(
                product_texts, 
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Create semantic embeddings for styles
            style_texts = (
                self.style_data['Style'].fillna('') + '. ' +
                self.style_data['Description'].fillna('') + '. ' +
                self.style_data['Common Clothing Items'].fillna('') + '.'
            ).str.lower().tolist()
            
            self.style_embeddings = self.semantic_model.encode(
                style_texts,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Cache embeddings for faster startup
            os.makedirs('Data', exist_ok=True)
            with open(product_embeddings_path, 'wb') as f:
                pickle.dump(self.product_embeddings, f)
            with open(style_embeddings_path, 'wb') as f:
                pickle.dump(self.style_embeddings, f)
        
        logger.info("Semantic search initialized")
    
    def search_products_by_text(self, query_text, max_results=10, budget_range=None):
        """Find products using semantic similarity search"""
        # Convert user query to semantic vector
        query_embedding = self.semantic_model.encode([query_text.lower()])
        
        # Calculate semantic similarity scores
        similarities = cosine_similarity(query_embedding, self.product_embeddings).flatten()
        
        # Get best matches
        top_indices = similarities.argsort()[::-1][:max_results * 3]
        
        # Build results with similarity scores
        results = self.catalog_data.iloc[top_indices].copy()
        results['similarity_score'] = similarities[top_indices]
        
        # Apply budget filter if specified
        if budget_range:
            min_price, max_price = budget_range
            results = results[
                (results['price'] >= min_price) & 
                (results['price'] <= max_price)
            ]
        
        # Filter for quality matches
        results = results[results['similarity_score'] > 0.1]
        
        return results.head(max_results)
    
    def search_products_by_category(self, categories, max_per_category=2, budget_range=None):
        """Find products by categories using semantic understanding"""
        all_results = []
        
        for category in categories:
            # Enhanced category query for better semantic matching
            enhanced_query = f"{category} clothing item fashion wear style"
            
            # Search using semantic similarity
            category_results = self.search_products_by_text(
                enhanced_query, 
                max_results=max_per_category * 3,
                budget_range=budget_range
            )
            
            filtered_results = category_results.head(max_per_category)
            all_results.append(filtered_results)
        
        # Combine all category results
        if all_results:
            return pd.concat(all_results, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def find_matching_styles(self, user_query, max_results=3):
        """Find outfit styles using semantic matching"""
        # Convert query to semantic vector
        query_embedding = self.semantic_model.encode([user_query.lower()])
        
        # Calculate semantic similarities with styles
        similarities = cosine_similarity(query_embedding, self.style_embeddings).flatten()
        
        # Get best style matches
        top_indices = similarities.argsort()[::-1][:max_results]
        
        results = self.style_data.iloc[top_indices].copy()
        results['similarity_score'] = similarities[top_indices]
        
        return results[results['similarity_score'] > 0.2]
    
    def classify_product_intent(self, query_text):
        """Automatically detect what type of products user wants"""
        query_embedding = self.semantic_model.encode([query_text.lower()])
        
        # Define product type embeddings
        product_types = {
            'clothing': 'shirt dress pants jacket sweater top bottom outerwear clothing apparel',
            'perfume': 'perfume fragrance scent eau de parfum cologne spray fragrance',
            'accessories': 'bag purse watch jewelry necklace bracelet ring earrings accessories',
            'shoes': 'shoes sneakers boots heels sandals footwear'
        }
        
        max_similarity = 0
        best_type = 'clothing'  # default
        type_scores = {}
        
        for ptype, description in product_types.items():
            type_embedding = self.semantic_model.encode([description])
            similarity = cosine_similarity(query_embedding, type_embedding)[0][0]
            type_scores[ptype] = similarity
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_type = ptype
        
        return {
            'product_type': best_type,
            'confidence': max_similarity,
            'all_scores': type_scores
        }
    
    def semantic_outfit_recommendation(self, query_text, max_results=6):
        """Create complete outfit recommendations using semantic understanding"""
        # Classify the intent first
        intent = self.classify_product_intent(query_text)
        
        if intent['product_type'] == 'perfume':
            return self.search_products_by_text(f"{query_text} perfume fragrance", max_results)
        
        # For clothing, create a complete outfit
        outfit_query = f"{query_text} complete outfit coordinated style fashion"
        
        # Get diverse items for a complete look
        results = self.search_products_by_text(outfit_query, max_results * 2)
        
        # Ensure variety in the outfit
        diverse_results = self._ensure_outfit_diversity(results, max_results)
        
        return diverse_results
    
    def _ensure_outfit_diversity(self, results, max_items):
        """Ensure outfit has diverse item types"""
        if results.empty:
            return results
        
        # Simple category classification
        categories = ['top', 'bottom', 'dress', 'jacket', 'accessories']
        category_keywords = {
            'top': ['shirt', 'blouse', 'top', 'sweater', 't-shirt'],
            'bottom': ['pants', 'jeans', 'skirt', 'shorts', 'trousers'],
            'dress': ['dress', 'gown'],
            'jacket': ['jacket', 'blazer', 'coat', 'cardigan'],
            'accessories': ['bag', 'belt', 'scarf', 'hat']
        }
        
        # Categorize items
        item_categories = []
        for _, item in results.iterrows():
            item_text = item['search_text'].lower()
            best_category = 'top'  # default
            
            for category, keywords in category_keywords.items():
                if any(keyword in item_text for keyword in keywords):
                    best_category = category
                    break
            
            item_categories.append(best_category)
        
        results['item_category'] = item_categories
        
        # Select diverse items
        diverse_items = []
        used_categories = set()
        
        # First pass: one item per category
        for _, item in results.iterrows():
            if item['item_category'] not in used_categories:
                diverse_items.append(item)
                used_categories.add(item['item_category'])
                if len(diverse_items) >= max_items:
                    break
        
        # Second pass: fill remaining slots
        if len(diverse_items) < max_items:
            for _, item in results.iterrows():
                if len(diverse_items) >= max_items:
                    break
                if not any((existing == item).all() for existing in diverse_items):
                    diverse_items.append(item)
        
        return pd.DataFrame(diverse_items[:max_items])
    
    # Helper methods for backward compatibility
    def get_catalog_data(self):
        """Return the product catalog"""
        return self.catalog_data
    
    def query_data(self, query):
        """Execute pandas query on catalog"""
        return self.catalog_data.query(query)
    
    def get_categories(self):
        """Get unique product categories"""
        return self.catalog_data['terms'].dropna().unique()
    
    def get_price_range(self):
        """Get min and max prices from catalog"""
        return {
            'min_price': self.catalog_data['price'].min(),
            'max_price': self.catalog_data['price'].max()
        }
    
    def extract_budget_from_text(self, text):
        """Extract budget information from text"""
        import re
        
        text = text.lower()
        
        # Look for explicit price mentions
        price_patterns = [
            r'\$(\d+)',
            r'(\d+)\s*dollars?',
            r'under\s*\$?(\d+)',
            r'less\s*than\s*\$?(\d+)',
            r'budget\s*of\s*\$?(\d+)',
            r'maximum\s*\$?(\d+)',
            r'max\s*\$?(\d+)'
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, text)
            if match:
                amount = int(match.group(1))
                return (0, amount)
        
        # Budget categories
        if any(word in text for word in ['cheap', 'budget', 'affordable', 'inexpensive']):
            return (0, 50)
        elif any(word in text for word in ['expensive', 'luxury', 'premium', 'high-end']):
            return (200, 1000)
        elif any(word in text for word in ['mid-range', 'moderate', 'reasonable']):
            return (50, 200)
        
        return None

# Create the API instance for import
fashion_api = FashionDataAPI() 