from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def evaluate_accuracy(original_text, generated_report):
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit and transform the original text and generated report
    tfidf_matrix = vectorizer.fit_transform([original_text, generated_report])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    # Convert similarity to percentage
    accuracy = similarity * 100
    
    return accuracy