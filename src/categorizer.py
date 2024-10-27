import json
from typing import Optional

from groq import Groq

from arxiv_paper import ArxivPaper, get_arxiv_papers_with_id
from prompts import CATEGORIZER_SYSTEM_PROMPT

MAX_SUB_CATEGORIES = 20


def get_categories_and_subcategories(
    summary: str,
    client: Optional[Groq] = None,
) -> tuple[str, list[str]]:
    """
    Get categories and sub-categories from the given summary.

    Parameters
    ----------
    summary : str
        The summary of the paper.

    Returns
    -------
    tuple[str, list[str]]
        The category and sub-categories of the paper.
    """

    if client is None:
        client = Groq()

    completion = client.chat.completions.create(
        model="llama-3.2-3b-preview",
        messages=[
            {
                "role": "system",
                "content": CATEGORIZER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": summary,
            },
        ],
        temperature=0,
        max_tokens=1024,
        top_p=1,
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )

    result = json.loads(completion.choices[0].message.content)

    return result["category"], result["sub_categories"][:MAX_SUB_CATEGORIES]


def get_categorized_papers_from_ids(paper_ids: list[str]) -> list[ArxivPaper]:
    """
    Get categorized papers with given IDs.

    Parameters
    ----------
    paper_ids : list[str]
        A list of arXiv paper IDs.

    Returns
    -------
    list[ArxivPaper]
        A list of ArxivPaper objects.
    """

    papers = get_arxiv_papers_with_id(paper_ids)

    for paper in papers:
        category, sub_categories = get_categories_and_subcategories(paper.summary)

        paper.category = category
        paper.sub_categories = sub_categories

    return papers
