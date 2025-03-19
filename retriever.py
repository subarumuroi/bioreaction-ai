# Retriever py is a simple program for using the Semantic Scholar API to fetch research papers based on a given query.
# an example is provided below to fetch papers related to E. coli fermentation medium.
# currently this is setup for just 5 papers (change limit = 5 to increase)

import requests
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import time

# Load a pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# FAISS index file
INDEX_FILE = "faiss_index.pkl"

def fetch_papers(query, limit=5):
    """
    Fetches research papers from Semantic Scholar based on the given query.
    
    Args:
        query (str): The search term (e.g., organism + bioreaction).
        limit (int): The number of papers to fetch.
    
    Returns:
        list: A list of dictionaries containing the paper title, abstract, and URL.
    """

    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&fields=title,abstract,url&limit={limit}"

    while True:
        
        response = requests.get(url)

        if response.status_code == 429:
            print("Rate limit hi. sleeping for 10 seconds before retrying...")
            time.sleep(10) # sleep for 10 seconds before retrying
            continue # retry the request

        if response.status_code == 200:
            papers = response.json().get("data",[])
            return [
                {
                    "title": paper.get("title", "No title"),
                    "abstract": paper.get("abstract", "No abstract available"),
                    "url": paper.get("url", "#")
                }
                for paper in papers
            ]
        else:
            print(f"Error fetching papers: {response.status_code}")
            return []
    time.sleep(5) #pause between api calls
    
def create_faiss_index(papers):
    """
    Creates a FAISS index from the research papers and saves it
    """
    abstracts = [paper["abstract"] for paper in papers]
    embeddings = model.encode(abstracts, convert_to_numpy = True)
    
    #test to see as API may be failing to call the abstracts
    if embeddings.size == 0:
        print("No valid embeddings found. Skipping FAISS index creation.")
        return
    
    # creat FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # save index and metadata
    with open(INDEX_FILE, "wb") as f:
        pickle.dump((index, papers), f)

    print("FAISS index created and saved.")

def search_papers(query, top_k = 3):
    """
    Searches FAISS index for the most relevant papers
    """
    with open(INDEX_FILE, "rb") as f:
        index, papers = pickle.load(f)

    query_embedding = model.encode([query], convert_to_numpy = True)
    distances, indices = index.search(query_embedding, top_k)

    results = [papers[i] for i in indices[0]]
    return results
   
# Example usage (run this file directly to test)
if __name__ == "__main__":
    # Fetch and store in FAISS
    query = "E. coli fermentation medium"
    papers = fetch_papers(query)
    create_faiss_index(papers)

    # Retrieve most relevant papers
    search_results = search_papers("optimal fermentation conditions for E. coli")
    for idx, paper in enumerate(papers):
        print(f"{idx+1}. {paper['title']}\n   {paper['url']}\n")