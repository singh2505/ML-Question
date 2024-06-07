import os
import json
import nltk
import requests
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nltk.download('punkt')

class BM25Retriever:
    def __init__(self, documents):
        self.documents = documents
        self.vectorizer = CountVectorizer()
        self.transformer = TfidfTransformer()
        self.tfidf_matrix = self.transformer.fit_transform(self.vectorizer.fit_transform(self.documents))

    def retrieve(self, query, top_n=5):
        query_vec = self.transformer.transform(self.vectorizer.transform([query]))
        cosine_sim = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(cosine_sim)[-top_n:]
        return [self.documents[i] for i in top_indices[::-1]]

class LangChainQA:
    def __init__(self, api_url, api_token):
        self.api_url = api_url
        self.api_token = api_token

    def query(self, question, context):
        headers = {"Authorization": f"Bearer {self.api_token}"}
        payload = {
            "inputs": {
                "question": question,
                "context": context
            }
        }
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
            return response.json().get('answer', "No answer found")
        except requests.exceptions.RequestException as e:
            return f"Error in API response: {e}"

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return [doc.get("articleBody", "") for doc in data]

def main(question):
    data_path = 'data/israel_hamas_war.json'

    data = load_data(data_path)

    retriever = BM25Retriever(data)
    retriever_results = retriever.retrieve(question, top_n=1)

    context = retriever_results[0] if retriever_results else ""

    # Ensure you have the correct API endpoint
    api_url = "https://api-inference.huggingface.co/models/deepset/bert-base-cased-squad2"
    api_token = "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"  # Replace with your actual API token
    langchain_qa = LangChainQA(api_url, api_token)
    answer = langchain_qa.query(question, context)

    print(f"Question: {question}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python qa_system.py <question>")
        sys.exit(1)
    question = sys.argv[1]
    main(question)
