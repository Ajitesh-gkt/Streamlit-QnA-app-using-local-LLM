# A test script to check if chromadb and document pre processing was working correctly
from llm_chains import load_vectordb, create_embeddings

if __name__ == "__main__":
    vector_db = load_vectordb(create_embeddings())
    output = vector_db.similarity_search("Big mac index")
    print(output)