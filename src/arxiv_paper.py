from typing import Optional

import arxiv
from pydantic import BaseModel


class ArxivPaper(BaseModel):
    """
    A class to represent an arXiv paper.

    Attributes
    ----------
    id : Optional[int]
        The ID of the paper.
    title : str
        The title of the paper.
    authors : list[str]
        The authors of the paper.
    url : str
        The URL of the paper.
    summary : str
        The summary of the paper.
    category : str
        The category of the paper.
    sub_categories : list[str]
        The sub-categories of the paper.
    is_favorite : bool
        A flag indicating whether the paper is a favorite or not.
    """

    id: Optional[int] = None
    title: str
    authors: list[str]
    url: str
    summary: str
    category: str
    sub_categories: list[str]
    is_favorite: bool = False


def get_arxiv_papers_with_id(ids: list[str]) -> list[ArxivPaper]:
    """
    Get arXiv papers with given IDs.

    Parameters
    ----------
    ids : list[str]
        A list of arXiv paper IDs.

    Returns
    -------
    list[ArxivPaper]
        A list of arXivPaper objects.
    """

    papers = []

    for paper_id in ids:
        paper = next(arxiv.Search(id_list=[paper_id]).results())

        papers.append(
            ArxivPaper(
                title=paper.title,
                authors=[author.name for author in paper.authors],
                url=paper.pdf_url,
                summary=paper.summary,
                category="",
                sub_categories=[],
            )
        )

    return papers
