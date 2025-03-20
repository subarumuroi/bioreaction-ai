import streamlit as st
from retriever import fetch_papers, create_faiss_index, search_papers
import re
from collections import Counter
import os

# Function to extract potential media mentions
def extract_medium(abstracts):
    medium_keywords = [
        'medium', 'broth', 'nutrient', 'co-factors', 'media', 
        "LB broth", "M9 medium", "minimal medium", "rich medium", 
        "nutrient broth", "tryptic soy broth", 'yeast extract', 'glucose medium'
    ]

    found_media = []
    
    for abstract in abstracts:
        for keyword in medium_keywords:
            if re.search(rf"\b{keyword}\b", abstract, re.IGNORECASE):
                found_media.append(keyword)
    
    return Counter(found_media)

# Streamlit UI
st.title('Bioreaction Medium Recommender')
st.markdown('Enter a query related to bioreactions, and this tool will suggest an ideal medium based on literature.')

query = st.text_input("Enter your query (e.g. 'E. coli fermentation medium')", "")

if st.button('Find Ideal Medium'):
    if query:
        st.write('Fetching papers...')
        papers = fetch_papers(query, limit=10)
        
        # debugging: print fetched papers
        st.write("Fetched papers:", [paper["title"] for paper in papers])

        if papers:
            # ensure faiss index is refreshed
            if os.path.exists("faiss_index.pkl"):
                os.remove("faiss_index.pkl")
            create_faiss_index(papers) # reuse Faiss index from retriever.py

            #debugging: Check how many papers are indexed
            st.write(f"Total papers indexed: {len(papers)}")
            
            search_results = search_papers(query, top_k= min(5, len(papers))) # Prevent overrequesting
            
            # debugging: check search results
            st.write(f"Search results returned {len(search_results)} papers.")

            # ensure abstracts are available
            abstracts = [paper['abstract'] for paper in search_results if paper.get('abstract')]

            if abstracts:
                st.write('Analyzing papers...')
                medium_counts = extract_medium(abstracts)
            
                if medium_counts:
                    recommended_medium = medium_counts.most_common(1)[0][0]
                    st.subheader(f'Recommended Medium: {recommended_medium}')
                    st.write("Based on mentions in the following papers:")
                    
                    for idx, paper in enumerate(search_results):
                        st.markdown(f"**{idx+1}. {paper['title']}**")
                        st.write(f"🔗 [Read more]({paper['url']})")
                        st.write("---")
                else:
                    st.warning("No specific medium detected. Try refining your query.")
            else: 
                st.warning("No abstracts avialable in the retrieved papers.")
        else:
            st.error("No papers found. Try a different query.")
    else:
        st.warning("Please enter a query.")
