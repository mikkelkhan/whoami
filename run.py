from src.docload import load_doc
from src.embeddings import EmbeddingDoc
from src.vectorstore import FaissvectorStore
from src.search import RAGSearch

if __name__ == "__main__":
    #docs = load_doc("data")
    store = FaissvectorStore("faiss_store")
    #store.build_from_documents(docs)
    store.load()
    rag_search = RAGSearch()
    query = "what is the tech stack in mevert app??"
    summary = rag_search.search_and_summarize(query,top_k=3)
    print("summary:",summary)
    #print(store.query("tech stack?",top_k=3))
             

    

