import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from src.embeddings import EmbeddingDoc

class FaissvectorStore:
    def __init__(self,persist_dir = "faiss_store",embedding_model_name="all-MiniLM-L6-v2", chunk_size=500,chunck_overlap = 100):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir,exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunck_overlap
        self.model_name = SentenceTransformer(embedding_model_name)
    
    def build_from_documents(self,documnets):
        emb_pip = EmbeddingDoc(model_name=self.embedding_model_name,chunk_size=self.chunk_size,chunck_overlap=self.chunk_overlap)
        chunks = emb_pip.chunk_doc(documnets)
        embeddings = emb_pip.embedding_chunks(chunks)
        metadatas  = metadatas = [{"text": chunk.page_content}for chunk in chunks]
        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        self.save()

    def add_embeddings(self, embeddings, metadatas = None):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
    

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")

    def search(self, query_embedding, top_k= 5):
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": idx, "distance": dist, "metadata": meta})
        return results

    def query(self, query_text, top_k= 5):
        print(f"[INFO] Querying vector store for: '{query_text}'")
        query_emb = self.model_name.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k)         