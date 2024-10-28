import logging
import os
import sqlite3

from dotenv import load_dotenv
from flask import Flask, g, jsonify, redirect, render_template, request, url_for

from categorizer import get_categorized_papers_from_ids
from embedding_generator import generate_embedding
from vector_db import VectorDB

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

load_dotenv()

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect("vector_db/arxiv_papers.db", check_same_thread=False)
        g.vector_db = VectorDB(dim=768, conn=g.db)
    return g.vector_db


@app.teardown_appcontext
def close_db(e=None):  # pylint: disable=unused-argument
    db = g.pop("db", None)
    if db is not None:
        db.close()


@app.route("/")
def index():
    db = get_db()
    papers = db.get_all_papers()
    return render_template("index.html", papers=papers)


@app.route("/add_papers", methods=["GET", "POST"])
def add_papers():
    if request.method == "POST":
        paper_ids = request.form.getlist("paper_ids")
        logging.debug(f"Received paper_ids: {paper_ids}")
        if not paper_ids:
            logging.error("No paper_ids provided")
            return jsonify({"error": "No paper_ids provided"}), 400

        papers = get_categorized_papers_from_ids(paper_ids)
        db = get_db()
        db.add_data(papers)
        return redirect(url_for("index"))
    return render_template("add_papers.html")


@app.route("/query", methods=["GET", "POST"])
def query():
    if request.method == "POST":
        query_text = request.form.get("query_text", "")
        k = int(request.form.get("k", 3))
        min_similarity = float(request.form.get("min_similarity", 0.6))

        if not query_text:
            return jsonify({"error": "No query_text provided"}), 400

        query_embeddings = generate_embedding(query_text)
        db = get_db()
        result_papers = db.query(
            query_vector=query_embeddings, k=k, min_similarity=min_similarity
        )

        return render_template(
            "query.html",
            query_text=query_text,
            results=[
                {
                    "id": x.id,
                    "title": x.title,
                    "authors": x.authors,
                    "url": x.url,
                    "summary": x.summary,
                    "category": x.category,
                    "sub_categories": x.sub_categories,
                    "is_favorite": db.is_favorite(x.id),
                }
                for x in result_papers
            ],
        )
    return render_template("query.html", results=[])


@app.route("/favorites")
def favorites():
    db = get_db()
    favorite_papers = db.get_favorite_papers()
    return render_template("favorites.html", papers=favorite_papers)


@app.route("/favorite/<int:paper_id>", methods=["POST"])
def favorite(paper_id):
    db = get_db()
    db.add_favorite(paper_id)
    return jsonify({"status": "success"})


@app.route("/unfavorite/<int:paper_id>", methods=["POST"])
def unfavorite(paper_id):
    db = get_db()
    db.remove_favorite(paper_id)
    return jsonify({"status": "success"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
