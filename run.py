from src.docload import load_doc
from src.embeddings import EmbeddingDoc
from src.vectorstore import FaissvectorStore

if __name__ == "__main__":
    #docs = load_doc("data")
    store = FaissvectorStore("faiss_store")
    store.load()
    print(store.query("What is Mevert?",top_k=5))

    

