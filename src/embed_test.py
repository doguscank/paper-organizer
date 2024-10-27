# -*- coding: utf-8 -*-
import os

from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

from embedding_generator import generate_embedding
from vector_db import VectorDB

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

load_dotenv()

if __name__ == "__main__":
    db = VectorDB(dim=768)

    query = "image gen"
    query_embeddings = generate_embedding(query)
    papers = db.get_all_papers()

    for paper in papers:
        print(f"Paper: {paper.title}")
        print(f"Category: {paper.category}")
        print(f"Sub-categories: {paper.sub_categories}")

        name_embed_dict = {paper.category: generate_embedding(paper.category)}

        for subcat in paper.sub_categories:
            name_embed_dict[subcat] = generate_embedding(subcat)

        for name, embed in name_embed_dict.items():
            similarity = cosine_similarity(query_embeddings, embed)
            print(f"{name} vs {query}: {similarity}")

        print("-" * 50)
