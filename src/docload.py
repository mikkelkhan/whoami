from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from pathlib import Path
from typing import List,Any


def load_doc(data_dir):
    doc_path = Path(data_dir).resolve()
    documents = []
    pdf_files = list(doc_path.glob("**/*.pdf"))
    for pdf_doc in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_doc))
            loaded = loader.load()
            documents.extend(loaded)
        except Exception as e:
            print(f"[error] failed top load {pdf_doc}: {e}")
    return documents



