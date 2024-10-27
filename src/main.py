import os

from dotenv import load_dotenv

from categorizer import get_categorized_papers_from_ids
from embedding_generator import generate_embedding
from vector_db import VectorDB

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

load_dotenv()

if __name__ == "__main__":
    arxiv_ids = ["2112.10752", "2410.05258", "2410.07073"]

    db = VectorDB(dim=768)

    papers = get_categorized_papers_from_ids(arxiv_ids)

    db.add_data(papers)

    query_embeddings = generate_embedding("image gen")
    result_papers = db.query(query_vector=query_embeddings, k=3, min_similarity=0.6)

    print([x.title for x in result_papers])
