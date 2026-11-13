"""
Person 3 - Ranked Search Module
Performs Vector Space Model (TF-IDF) and Cosine Similarity ranking
on the clean dataset produced by Preprocessing.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_dataset(path="dataset/documents.csv"):
    """Load the preprocessed dataset csv file into a pandas DataFrame."""
    df = pd.read_csv(path)
    # Ensure there are no missing or null values in the processed text column
    df['processed_text'] = df['processed_text'].fillna('')
    return df

class RankedSearch:
    def __init__(self, csv_path="dataset/documents.csv"):
        self.df = load_dataset(csv_path)
        self.documents = self.df['processed_text'].tolist()
        
        # Initialize TF-IDF Vectorizer and build vector space model for all documents
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform(self.documents)

    def search(self, query, top_k=5):
        """Perform ranked search based on Cosine Similarity against the query."""
        # Transform the query text into the same vector space model dimensions
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity scores between the query vector and all document vectors
        cosine_scores = cosine_similarity(query_vector, self.doc_vectors).flatten()
        
        # Get the indices of top scoring documents in descending order (highest similarity first)
        top_indices = cosine_scores.argsort()[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            score = cosine_scores[idx]
            # Skip documents that have zero relevance score
            if score == 0:
                continue
            
            # Clean doc_id and convert it to integer to prevent formatting errors
            raw_id = str(self.df.loc[idx, 'doc_id'])
            doc_id = int(raw_id.replace('DOC', '').strip())
            
            title = self.df.loc[idx, 'title']
            category = self.df.loc[idx, 'category']
            
            results.append({
                'rank': rank,
                'doc_id': doc_id,
                'score': score,
                'title': title,
                'category': category
            })
            
        return results

def show_ranked_results(query, results):
    """Print the ranked search results in a clean, readable format."""
    print(f"\n--- Ranked Search Results for Query: '{query}' ---")
    if not results:
        print("No relevant documents found.")
        return
    for res in results:
        print(f"Rank {res['rank']} | Score: {res['score']:.4f} | ID: D{res['doc_id']:03d} | Category: {res['category']}")
        print(f"    Text: {res['title']}\n")

if __name__ == "__main__":
    engine = RankedSearch("dataset/documents.csv")
    test_query = "how to replace my lost card"
    results = engine.search(test_query, top_k=5)
    show_ranked_results(test_query, results)