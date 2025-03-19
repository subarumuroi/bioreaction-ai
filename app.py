import streamlit as st
from retriever import fetch_papers, create_faiss_index, search_papers

# Title and description
st.title('Paper Search Engine')
st.markdown('Enter a query related to bioreactions, and this tool will fetch relevant research papers for you.')


#User input

query = st.text_input("Enter your query (e.g. 'E. coli fermentation medium')", "")

if st.button('Search Papers'):
    if query:
        #Fetch papers
        st.write('Fetching papers...')
        papers = fetch_papers(query, limit=5)

        if papers:
            # Create FAISS index
            create_faiss_index(papers)

            # Retrieve most relevant papers
            search_results = search_papers(query, top_k=3)

            st.subheader('Top relevant papers:')
            for idx, paper in enumerate(search_results):
                st.markdown(f"**{idx+1}. {paper['title']}**")
                st.write(f"🔗 [Read more]({paper['url']})")
                st.write("---")
        else:
            st.error("No papers found. Try a different query.")
    else:
        st.warning("Please enter a query.")

# run streamlit app with: 'streamlit run app.py'