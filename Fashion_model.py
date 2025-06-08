# Fashion Data API - Manages product data and RAG search functionality
import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FashionDataAPI:
    """Main class for handling fashion product data and intelligent search"""
    
    def __init__(self):
        # Load product and style data from CSV files
        csv_path = os.path.join('Data', 'store_zara.csv')
        style_csv_path = os.path.join('Data', 'Common_Outfit_Styles.csv')
        
        # Load Zara product catalog
        logger.info("Loading fashion catalog data...")
        self.catalog_data = pd.read_csv(csv_path)
        self._prepare_catalog_data()
        
        # Load outfit style definitions
        logger.info("Loading style data...")
        self.style_data = pd.read_csv(style_csv_path)
        
        # Set up AI search capabilities
        self._initialize_vectorizers()
        
        logger.info(f"Loaded {len(self.catalog_data)} products and {len(self.style_data)} styles")
    
    def _prepare_catalog_data(self):
        """Clean and prepare product data for search"""
        # Remove unnecessary columns (keep 'url' for clickable links)
        columns_to_drop = ["brand", "sku", "currency", "scraped_at", "image_downloads", "error"]
        self.catalog_data.drop(columns=[col for col in columns_to_drop if col in self.catalog_data.columns], inplace=True)
        
        # Fix price format
        self.catalog_data['price'] = pd.to_numeric(self.catalog_data['price'], errors='coerce')
        
        # Create searchable text by combining all product info
        self.catalog_data['search_text'] = (
            self.catalog_data['name'].fillna('') + ' ' +
            self.catalog_data['description'].fillna('') + ' ' +
            self.catalog_data['terms'].fillna('') + ' ' +
            self.catalog_data['section'].fillna('')
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
    
    def _initialize_vectorizers(self):
        """Set up AI search using TF-IDF vectorization"""
        logger.info("Initializing TF-IDF vectorizers...")
        
        # Product search AI
        self.product_vectorizer = TfidfVectorizer(
            max_features=5000,      # Vocabulary size
            stop_words='english',   # Remove common words
            lowercase=True,         # Normalize case
            ngram_range=(1, 2)      # Use 1-2 word phrases
        )
        
        # Train on product descriptions
        self.product_tfidf_matrix = self.product_vectorizer.fit_transform(
            self.catalog_data['search_text'].fillna('')
        )
        
        # Style matching AI
        self.style_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)
        )
        
        # Train on style descriptions
        style_text = (
            self.style_data['Style'].fillna('') + ' ' +
            self.style_data['Description'].fillna('') + ' ' +
            self.style_data['Common Clothing Items'].fillna('')
        ).str.lower()
        
        self.style_tfidf_matrix = self.style_vectorizer.fit_transform(style_text)
        
        logger.info("TF-IDF vectorizers initialized successfully")
    
    def search_products_by_text(self, query_text, max_results=10, budget_range=None):
        """
        Find products using AI text similarity search
        """
        # Convert user query to AI vector
        query_vector = self.product_vectorizer.transform([query_text.lower()])
        
        # Calculate similarity scores with all products
        similarities = cosine_similarity(query_vector, self.product_tfidf_matrix).flatten()
        
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
        
        # Remove very poor matches
        results = results[results['similarity_score'] > 0.01]
        
        return results.head(max_results)
    
    def search_products_by_category(self, categories, max_per_category=2, budget_range=None):
        """
        Find products by specific categories (shirts, pants, etc.)
        """
        all_results = []
        
        for category in categories:
            # Search for this category
            category_results = self.search_products_by_text(
                category, 
                max_results=max_per_category * 3,
                budget_range=budget_range
            )
            
            # Prefer exact category matches
            category_mask = category_results['terms'].str.contains(
                category.lower(), case=False, na=False
            )
            
            filtered_results = category_results[category_mask].head(max_per_category)
            
            # Fill with similar items if needed
            if len(filtered_results) < max_per_category:
                remaining = max_per_category - len(filtered_results)
                fuzzy_results = category_results[~category_mask].head(remaining)
                filtered_results = pd.concat([filtered_results, fuzzy_results])
            
            all_results.append(filtered_results)
        
        # Combine all category results
        if all_results:
            return pd.concat(all_results, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def find_matching_styles(self, user_query, max_results=3):
        """
        Find outfit styles that match user description
        """
        # Convert query to style vector
        query_vector = self.style_vectorizer.transform([user_query.lower()])
        
        # Calculate style similarities
        similarities = cosine_similarity(query_vector, self.style_tfidf_matrix).flatten()
        
        # Get best style matches
        top_indices = similarities.argsort()[::-1][:max_results]
        
        results = self.style_data.iloc[top_indices].copy()
        results['similarity_score'] = similarities[top_indices]
        
        # Only return good matches
        return results[results['similarity_score'] > 0.05]
    
    # Helper methods for backward compatibility
    def get_catalog_data(self):
        """Return the product catalog"""
        return self.catalog_data
    
    def query_data(self, query):
        """Execute pandas query on catalog"""
        return self.catalog_data.query(query)
    
    def get_categories(self):
        """Get all available product categories"""
        return self.catalog_data['terms'].unique()
    
    def get_price_range(self):
        """Get catalog price statistics"""
        return {
            'min_price': self.catalog_data['price'].min(),
            'max_price': self.catalog_data['price'].max(),
            'avg_price': self.catalog_data['price'].mean()
        }
    
    def extract_budget_from_text(self, text):
        """
        Extract budget information from user text
        Examples: "under $100", "between $50 and $200", "less than 150"
        """
        import re
        
        text = text.lower()
        
        # Pattern: "under $X", "less than $X", "below $X"
        under_pattern = r'(?:under|less than|below|max)\s*\$?(\d+)'
        under_match = re.search(under_pattern, text)
        if under_match:
            max_price = float(under_match.group(1))
            return (0, max_price)
        
        # Pattern: "between $X and $Y"
        between_pattern = r'between\s*\$?(\d+)\s*(?:and|to|-)\s*\$?(\d+)'
        between_match = re.search(between_pattern, text)
        if between_match:
            min_price = float(between_match.group(1))
            max_price = float(between_match.group(2))
            return (min_price, max_price)
        
        # Pattern: "around $X", "about $X"
        around_pattern = r'(?:around|about|approximately)\s*\$?(\d+)'
        around_match = re.search(around_pattern, text)
        if around_match:
            target_price = float(around_match.group(1))
            # ±20% range
            min_price = target_price * 0.8
            max_price = target_price * 1.2
            return (min_price, max_price)
        
        # Pattern: "over $X", "more than $X", "above $X"
        over_pattern = r'(?:over|more than|above|min)\s*\$?(\d+)'
        over_match = re.search(over_pattern, text)
        if over_match:
            min_price = float(over_match.group(1))
            return (min_price, float('inf'))
        
        # No budget specified
        return None

# Create global instance for the application
fashion_api = FashionDataAPI()

# Example usage of the API
# result = fashion_api.query_data("your_query_here")
# print(result)

# You can add additional processing or database creation logic here
