# Content-Based News Article Recommendation System

A Streamlit-based web application that provides personalized news article recommendations based on user queries using natural language processing, machine learning techniques and transformers.

## Features

- Text-based search functionality for news articles
- Content-based recommendation using BERT embeddings
- Interactive web interface built with Streamlit
- Preprocessing of text data using NLTK
- Similarity-based article matching

## Tech Stack

- Python 3.11+
- Streamlit
- Sentence-Transformers (BERT)
- NLTK
- Pandas
- Scikit-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/RawatRahul14/Recommendation_System.git
cd Recommendation_System
```

2. Create a virtual environment and activate it:
```bash
python -m venv .venv
.venv/Scripts/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The system uses a news article dataset containing the following columns:
- Article: The main content of the news article
- Date: Publication date
- Heading: Article headline
- NewsType: Category of the news article

The dataset is automatically downloaded when running the application for the first time.

## Project Structure

```
Recommendation_System/
├── app.py                           # Streamlit web application
├── requirements.txt                 # Project dependencies
├── recommendation_system.ipynb      # Development notebook
├── Dataset/                         # Data directory
│   ├── data.csv                     # News articles dataset
│   └── embeddings.pkl               # Pre-computed BERT embeddings
└── utils/                           # Utility functions
    ├── __init__.py                  # Package initializer
    ├── common.py                    # Common utility functions
    └── recommendation.py            # Recommendation system logic
```
## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Enter your search query in the text input field

4. The system will display the top 5 most relevant news articles based on your query

## How It Works

1. **Data Preprocessing**:
   - Text cleaning and normalization
   - Stopword removal
   - Stemming using Porter Stemmer

2. **Embedding Generation**:
   - Uses the `all-MiniLM-L6-v2` BERT model
   - Generates embeddings for article headlines

3. **Recommendation**:
   - Converts user query into embeddings
   - Calculates cosine similarity with article embeddings
   - Returns top matching articles

## Development

To modify or extend the system, refer to the `recommendation_system.ipynb` notebook which contains the step-by-step development process and implementation details.

## Dependencies

- pandas: Data manipulation and analysis
- scikit-learn: Machine learning utilities
- sentence_transformers: BERT model implementation
- nltk: Natural language processing tools
- streamlit: Web application framework