# Retriever py is a simple program for using the Semantic Scholar API to fetch research papers based on a given query.
# an example is provided below to fetch papers related to E. coli fermentation medium.
# currently this is setup for just 5 papers (change limit = 5 to increase)

import requests
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import time
import os

# Load a pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# FAISS index file and metadata file
INDEX_FILE = "faiss_index.pkl"
PAPERS_FILE = "papers.pkl"

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

    retry_delay = 2 # Initial delay in seconds for exponentail backoff
    max_retries =5

    for attempt in range(max_retries):
        response = requests.get(url)
    
        if response.status_code == 429:
            print(f"Rate limit hit. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2 # exponential backoff
            continue # retry the request

        elif response.status_code == 200:
            papers = response.json().get("data",[])
            return [
                {
                    "title": paper.get("title", "No title"),
                    "abstract": paper.get("abstract", "No abstract available"),
                    "url": paper.get("url", "#")
                }
                for paper in papers if paper.get("abstract") # Ensure abstract exists
            ]
        else:
            print(f"Error fetching papers: {response.status_code}")
            return []
    
    print("Failed to fetch papers after {max_retries} attempts.")

def load_existing_index():
    """
    Loads an existing FAISS index and metadata file if they exist.
    """
    if os.path.exists(INDEX_FILE) and os.path.exists(PAPERS_FILE):
        with open(INDEX_FILE, "rb") as f:
            index = pickle.load(f)
        with open(PAPERS_FILE, "rb") as f:
            papers = pickle.load(f)
        return index, papers
    else:
        return None, []

def create_faiss_index(new_papers):
    """
    Creates or updates the FAISS index from the research papers and saves it
    """
    index, existing_papers = load_existing_index()

    # Combine new and old papers, avoiding duplicates
    paper_titles = {paper["title"] for paper in existing_papers}
    papers = existing_papers + [p for p in new_papers if p["title"] not in paper_titles]

    
    abstracts = [paper["abstract"] for paper in papers]
    embeddings = model.encode(abstracts, convert_to_numpy = True)
    
    if index is None:
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings) # First time adding embeddings
    else:
        existing_embeddings = index.ntotal
        if existing_embeddings > 0:
            print(f"Existing FAISS index size: {existing_embeddings}")

            # check is new embeddings are already in the index
            _, duplicates_indices = index.search(embeddings, 1) # Find nearest existing match
            uniuqe_indices = [i for i in range(len(embeddings)) if duplicates_indices[i][0] >= existing_embeddings]

            if uniuqe_indices:
                new_embeddings = embeddings[uniuqe_indices]
                index.add(new_embeddings) # add only unique embeddings
      

    # save updated index and metadata
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(index, f)
    with open(PAPERS_FILE, "wb") as f:
        pickle.dump(papers, f)

    print(f"FAISS index updated with {len(new_papers)} new papers. Total papers: {len(papers)}")

def search_papers(query, top_k = 3):
    """
    Searches FAISS index for the most relevant papers
    """
    index, papers = load_existing_index()
    if index is None or len(papers) == 0:
        print("No FAISS index found. Run 'create_faiss_index()' first.")
        return []
    query_embedding = model.encode([query], convert_to_numpy = True)
    actual_k = min(top_k, len(papers))   
    distances, indices = index.search(query_embedding, actual_k)

    results = [papers[i] for i in indices[0] if i < len(papers)]
    return results
   
# Example usage (run this file directly to test)
if __name__ == "__main__":
    # Fetch and store in FAISS
    query = "E. coli fermentation medium"
    new_papers = fetch_papers(query)
    create_faiss_index(new_papers)

    # Retrieve most relevant papers
    search_results = search_papers("optimal fermentation conditions for E. coli")
    for idx, paper in enumerate(search_results):
        print(f"{idx+1}. {paper['title']}\n   {paper['url']}\n")