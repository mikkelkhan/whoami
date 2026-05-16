import os

from langchain_groq import ChatGroq

from src.vectorstore import FaissvectorStore


class RAGSearch:

    def __init__(
        self,
        persist_dir="faiss_store",
        embedding_model="all-MiniLM-L6-v2",
        llm_model="llama-3.3-70b-versatile"
    ):

        self.vectorstore = FaissvectorStore(
            persist_dir,
            embedding_model
        )

        faiss_path = os.path.join(
            persist_dir,
            "faiss.index"
        )

        meta_path = os.path.join(
            persist_dir,
            "metadata.pkl"
        )

        if not (
            os.path.exists(faiss_path)
            and
            os.path.exists(meta_path)
        ):

            from docload import load_all_documents

            docs = load_all_documents("data")

            self.vectorstore.build_from_documents(docs)

        else:

            self.vectorstore.load()

        groq_api_key = ""

        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=llm_model,
            temperature=0
        )

        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(
        self,
        query: str,
        top_k: int = 3
    ) -> str:

        results = self.vectorstore.query(
            query,
            top_k=top_k
        )

        texts = []

        for r in results:

            if r["metadata"]:

                texts.append(
                    r["metadata"].get("text", "")
                )

        context = "\n\n".join(texts)

        if not context:

            return "No relevant documents found."

       

        prompt = f"""
You are a helpful AI assistant.

Answer ONLY from the provided context.

If the answer is not found in the context,
say:
"I could not find the answer in the documents."

Context:
{context}

Question:
{query}

Answer:
"""


        response = self.llm.invoke(prompt)

        return response.content