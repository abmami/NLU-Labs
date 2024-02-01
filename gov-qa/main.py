import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import json
from nltk.stem.snowball import SnowballStemmer
import itertools
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from transformers import pipeline

nlp = pipeline('question-answering', model='etalab-ia/camembert-base-squadFR-fquad-piaf', tokenizer='etalab-ia/camembert-base-squadFR-fquad-piaf')


stop_words = set(stopwords.words('french'))
stemmer = SnowballStemmer(('french'))
punctuations = set(string.punctuation)


def process_tokens(words):
    tokens = [w for w in words if not w.lower() in stop_words and w.lower() not in punctuations]
    tokens = [stemmer.stem(token) for token in tokens]
    tokens = [t for t in  tokens if any(c.isnumeric() for c in t)==False]

    return tokens

def process_document(document):
    words = document.split(' ')
    words = process_tokens(words)
    return ' '.join(words)



def get_documents(json_file):
    with open(f'../data/{json_file}', encoding='UTF-8') as f:
        data = json.load(f)
        corpus = [item['content'] for item in data]
        return corpus
        


def get_relevant_doc(query, documents):
    processed_documents = [process_document(d) for d in documents]
    query = process_document(query)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_documents)
    query_vector = vectorizer.transform([query])

    cosine_similarities = np.dot(tfidf_matrix, query_vector.T)
    best_document_index = cosine_similarities.argmax()
    best_document = documents[best_document_index]
    return best_document



if __name__ == '__main__':
    documents = get_documents('finance_glob.json')
    query = input('Enter your query: ')
    doc = get_relevant_doc(query, documents)
    
    result = nlp({
    'question': query,
    'context': doc
    })

    print(result['answer'])