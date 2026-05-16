from src.docload import load_doc
from src.embeddings import EmbeddingDoc
from src.vectorstore import FaissvectorStore
from src.search import RAGSearch
from flask import Flask, render_template, request, jsonify



app = Flask(__name__)

rag = RAGSearch()


@app.route("/")
def home():
    return render_template("chat_ui.html")


@app.route("/chat", methods=["POST"])
def chat():

    data = request.json

    query = data.get("message")

    answer = rag.search_and_summarize(query)

    return jsonify({
        "answer": answer
    })


if __name__ == "__main__":
    app.run(debug=True)
             

    

