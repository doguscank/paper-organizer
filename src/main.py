from dotenv import load_dotenv
from categorizer import get_categorized_papers
from arxiv_paper import ArxivPaper
from vector_db import VectorDB
from embedding_generator import generate_embedding

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

load_dotenv()

if __name__ == "__main__":
    arxiv_ids = ["2112.10752", "2410.05258", "2410.07073"]

    db = VectorDB(dim=768)

    papers = get_categorized_papers(arxiv_ids)

    db.add_data(papers)

    query_embeddings = generate_embedding("Diffusion Models")
    result_papers = db.query(query_vector=query_embeddings, k=60)

    print([x.title for x in result_papers])
