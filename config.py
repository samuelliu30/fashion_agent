# Fashion Agent Configuration
# Switch between different AI search methods

# SEARCH METHOD OPTIONS:
# 'tfidf'    - Traditional TF-IDF vectorization (faster, good for exact matches)
# 'semantic' - Modern semantic embeddings (smarter, better context understanding)

SEARCH_METHOD = 'semantic'  # Using advanced semantic embeddings for better context understanding

# Other configuration options
DEFAULT_MAX_RESULTS = 6
CACHE_EMBEDDINGS = True
DEBUG_LOGGING = True

# Model configurations
TFIDF_CONFIG = {
    'max_features': 5000,
    'ngram_range': (1, 2),
    'stop_words': 'english'
}

SEMANTIC_CONFIG = {
    'model_name': 'all-MiniLM-L6-v2',  # Fast and efficient
    # Alternative models:
    # 'all-mpnet-base-v2'              # More accurate but slower
    # 'paraphrase-multilingual'        # Multi-language support
    'similarity_threshold': 0.1
} 