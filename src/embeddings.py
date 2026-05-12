from src.docload import load_doc
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


class EmbeddingDoc:
    def __init__(self,model_name="all-MiniLm-L6-v2", chunk_size=500,chunck_overlap = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunck_overlap
        self.model_name = SentenceTransformer(model_name)
    
    def chunk_doc(self,documents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function = len,
            separators=["\n\n","\n",""," "]
            

        )
        chunks = splitter.split_documents(documents)
        return chunks
    def embedding_chunks(self,chunks):
        texts = [chunk.page_content for chunk in chunks]
        embedding = self.model_name.encode(texts,show_progress_bar=True)
        return embedding
