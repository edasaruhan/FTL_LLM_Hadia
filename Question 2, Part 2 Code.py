import faiss
from sklearn.preprocessing import normalize

# Define function to build and search the vector database
def build_search_engine(embeddings, dim):
    # Normalize embeddings
    embeddings = normalize(embeddings)
    
    # Build FAISS index
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def search_query(query, index, embedder):
    query_embedding = embedder.embed([query])
    query_embedding = normalize(query_embedding)
    _, indices = index.search(query_embedding, k=5)  # Top 5 results
    return indices

# Example documents
documents = [
    "Global warming is a major issue facing the planet.",
    "The development of artificial intelligence is rapid.",
    "Education is crucial for economic development.",
    "Renewable energy sources are essential for sustainable growth.",
    "Healthcare improvements can lead to a better quality of life."
]

# Build the search engine with document embeddings
bert_embedder = BERTEmbedder()
doc_embeddings = bert_embedder.embed(documents)
index = build_search_engine(doc_embeddings, doc_embeddings.shape[1])

# Search for a query
query = "What are the benefits of renewable energy?"
indices = search_query(query, index, bert_embedder)

print("\nSearch Results:")
for i in indices[0]:
    print(documents[i])