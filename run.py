from src.docload import load_doc
from src.embeddings import EmbeddingDoc

if __name__ == "__main__":
    docs = load_doc("data")
    chunks = EmbeddingDoc().chunk_doc(docs)
    chunks_vector = EmbeddingDoc().embedding_chunks(chunks)

    print(chunks_vector)

