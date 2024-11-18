import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_data(input_file):
    with open(input_file, 'r') as f:
        text = f.read()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]
    
    # Join the tokens back into a string
    preprocessed_text = ' '.join(filtered_tokens)
    
    return preprocessed_text