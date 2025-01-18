from utils.common import preprocess_text
from sklearn.metrics.pairwise import cosine_similarity

def recommend_articles_from_search(query, dataset, embeddings, model, num_recommendations = 5):
    # Preprocess Query
    query = preprocess_text(query)

    # Encode the query
    query_embds = model.encode([query], convert_to_tensor = True)

    # Calculating Similarity
    similarities = cosine_similarity(query_embds.reshape(1, -1), embeddings)
    similarities = similarities.flatten()

    # Get the indices of the top recommendations
    top_indices = similarities.argsort()[-num_recommendations:][::-1]

    # Select the recommended articles
    recommended_articles = dataset.iloc[top_indices][["Heading", "NewsType", "Article", "Date"]]

    return recommended_articles