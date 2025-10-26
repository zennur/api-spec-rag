from api_docs_vectorizer import APIDocsVectorStore

# Create and populate vector store
vectorizer = APIDocsVectorStore()
vectorizer.load_documents('./api_docs_test')
vectorizer.create_embeddings()
vectorizer.build_index()

# Save for reuse
vectorizer.save('./saved_vector_store')

# Later, load and search
vectorizer_loaded = APIDocsVectorStore()
vectorizer_loaded.load('./saved_vector_store')

# Perform semantic search
results = vectorizer_loaded.search(
    "How to handle rate limiting?",
    top_k=5
)

# Display results
for chunk, metadata, score in results:
    print(f"\nRelevance Score: {score:.4f}")
    print(f"Source: {metadata['title']}")
    print(f"Content: {chunk}\n")