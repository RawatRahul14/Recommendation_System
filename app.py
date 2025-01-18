from utils.recommendation import recommend_articles_from_search
from utils.common import load_df, load_embds, get_model
import streamlit as st
import pandas as pd

# Load data, embeddings and model
dataset = load_df()
embeddings = load_embds()
model = get_model()

st.title("Content Based Recommendation")

query = st.text_input("Enter your query here...")

if query:
    # get recommendation
    recommended_articles = recommend_articles_from_search(query, dataset, embeddings, model, num_recommendations = 5)

    st.subheader("\n Top News Articles")
    st.dataframe(recommended_articles)

else:
    st.write("Please enter your query")