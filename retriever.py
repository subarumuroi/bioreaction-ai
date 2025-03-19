# Retriever py is a simple program for using the Semantic Scholar API to fetch research papers based on a given query.
# an example is provided below to fetch papers related to E. coli fermentation medium.
# currently this is setup for just 5 papers (change limit = 5 to increase)

import requests
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

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

    response = requests.get(url)

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
    
# Example usage (run this file directly to test)
if __name__ == "__main__":
    query = "E.Coli fermentation medium"
    papers = fetch_papers(query)
    for idx, paper in enumerate(papers):
        print(f"{idx+1}. {paper['title']}\n   {paper['url']}\n")