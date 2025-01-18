import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer

nltk.download("stopwords")
nltk.download("punkt")

stop_words = set(stopwords.words("English"))
stemmer = PorterStemmer()

def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Tokenize
    words = text.split()

    # Remove stopwords and apply stemming
    preprocessed_text = [stemmer.stem(word) for word in words if word not in stop_words]

    return " ".join(preprocessed_text)


def load_df(path = "Dataset/data.csv"):
    dataset = pd.read_csv(path, encoding = "ISO-8859-1")
    return dataset

def load_embds(path = "Dataset/embeddings.pkl"):
    with open(path, "rb") as file:
        embeddings = pickle.load(file)

    return embeddings

def get_model(model_name = "all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)

    return model