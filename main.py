import pandas as pd
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import nltk
from sklearn.metrics.pairwise import cosine_similarity

input_file = 'legal_text_classification.csv' 
output_file = 'summarized_cases.csv'

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

df = pd.read_csv(input_file)

if 'case_text' not in df.columns:
    raise ValueError("CSV file must contain a 'case_text' column.")

def custom_extractive_summary(text, ratio=0.2):
    # Handle empty or invalid text
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    
    # Handle cases with very few sentences
    if len(sentences) <= 3:
        return " ".join(sentences)
    
    # Clean sentences (remove stopwords)
    clean_sentences = [" ".join([word for word in sentence.lower().split() if word not in stop_words]) 
                      for sentence in sentences]
    
    # Create TF-IDF vectorizer and compute sentence vectors
    vectorizer = TfidfVectorizer()
    sentence_vectors = vectorizer.fit_transform(clean_sentences)
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(sentence_vectors)
    
    # Calculate sentence scores based on similarity
    sentence_scores = np.sum(similarity_matrix, axis=1)
    
    # Determine number of sentences for the summary
    num_sentences = max(1, int(len(sentences) * ratio))
    
    # Get top sentence indices (sorted by score)
    ranked_sentences = sorted(((sentence_scores[i], i) for i in range(len(sentences))), reverse=True)
    top_sentence_indices = [ranked_sentences[i][1] for i in range(num_sentences)]
    
    # Sort indices to preserve original sentence order
    top_sentence_indices.sort()
    
    # Generate summary
    summary = " ".join([sentences[i] for i in top_sentence_indices])
    
    return summary

# Apply the custom summarization to each case text
df['summary'] = df['case_text'].apply(lambda x: custom_extractive_summary(str(x)))

# Save the results
df.to_csv(output_file, index=False)

print(f"Output saved to {output_file}")
