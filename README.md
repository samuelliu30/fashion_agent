# Fashion Agent - Your AI Personal Stylist

A smart fashion recommendation system that actually understands what you're asking for. Built with Flask, OpenAI, and real Zara product data.

## What it does

Ask for fashion advice in plain English, and get personalized recommendations with real products you can actually buy. The system understands context, budgets, and different styles.

**Examples:**
- "I need a casual outfit for a date under $100"
- "Something professional for a business meeting"
- "I want an elegant dress for work"

## Quick Setup

1. **Clone and setup**
   ```bash
   git clone <your-repo>
   cd fashion_agent
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Add your OpenAI key**
   ```bash
   echo "OPENAI_API_KEY=your_key_here" > .env
   ```

3. **Run it**
   ```bash
   python app.py
   ```

4. **Open browser**
   Go to `http://localhost:5000`

## How it works

### Smart conversation handling
- Uses GPT-3.5 to understand what you're actually asking for
- Handles follow-up questions and clarifications
- Remembers conversation context

### Two search methods

**TF-IDF (default)** - Fast and reliable
- Quick startup
- Good for exact matches
- Lower memory usage

**Semantic search** - Better understanding
- Understands context and style
- Slower first startup
- Switch by changing `SEARCH_METHOD = 'semantic'` in `config.py`

### Real product data
- 3,065 actual Zara products with current prices
- Automatic image downloading and caching
- Clickable links to buy products
- Budget-aware filtering

## Key features

- **Context awareness**: "I'm a man" → shows men's clothing
- **Budget extraction**: Understands "$100 budget" or "something cheap"
- **Style matching**: 13 different fashion styles from casual to formal
- **Image handling**: Downloads and caches product photos
- **Conversation memory**: Maintains chat history
- **Intent classification**: Distinguishes fashion requests from casual chat

## Project structure

```
fashion_agent/
├── app.py                    # Flask web app
├── Agent.py                  # Main AI logic and conversation handling
├── Fashion_model.py          # TF-IDF search system
├── Fashion_model_semantic.py # Semantic search system
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── Data/                     # Product and style data
├── static/                   # Web assets and cached images
└── templates/                # HTML templates
```

## Tech stack

- **Backend**: Flask, Python
- **AI**: OpenAI GPT-3.5-turbo
- **Search**: TF-IDF or Sentence Transformers
- **Data**: Pandas, NumPy, scikit-learn
- **Frontend**: HTML, CSS, JavaScript

## Search method switching

Change between TF-IDF and semantic search by editing `config.py`:

```python
# Fast and reliable (default)
SEARCH_METHOD = 'tfidf'

# Better natural language understanding
SEARCH_METHOD = 'semantic'
```

**TF-IDF**: Great for exact word matches, fast startup
**Semantic**: Better context understanding, slower first startup

For semantic search, install additional dependency:
```bash
pip install sentence-transformers==2.2.2
```

## Common issues

**NumPy compatibility**: If you get NumPy 2.x errors:
```bash
pip install "numpy<2.0"
```

**Missing dependencies**: Make sure all packages are installed:
```bash
pip install -r requirements.txt
```

**OpenAI errors**: Check your API key in the `.env` file

## Development notes

The system went through several iterations:
- Started with basic regex-based intent detection
- Upgraded to AI-powered intent classification for better accuracy
- Added semantic search option for improved natural language understanding
- Fixed various edge cases around budget parsing and product filtering


