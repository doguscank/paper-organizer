from dotenv import load_dotenv
from categorizer import get_categorized_papers

load_dotenv()


def main(arxiv_ids: list[str]) -> None:
    papers = get_categorized_papers(arxiv_ids)

    for paper in papers:
        print(paper)


if __name__ == "__main__":
    arxiv_id = "2112.10752"

    main(arxiv_ids=[arxiv_id])
