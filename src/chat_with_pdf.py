import logging
from pathlib import Path
from typing import List, Tuple

import fitz
import requests
from groq import Groq

from arxiv_paper import ArxivPaper
from embedding_generator import find_distance_L2, generate_embedding
from prompts import QA_SYSTEM_PROMPT


def _download_paper_pdf(paper: ArxivPaper) -> Path:
    """
    Download the PDF of the given paper.

    Parameters
    ----------
    paper : ArxivPaper
        The paper to download the PDF for.

    Returns
    -------
    Path
        The path to the downloaded PDF file.
    """

    pdf_url = paper.url
    pdf_path = Path(f"pdfs/{paper.id}.pdf")

    if not pdf_path.exists():
        pdf_path.parent.mkdir(parents=True, exist_ok=True)

        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()

        with open(pdf_path, "wb") as f:
            f.write(response.content)

    return pdf_path


def parse_pdf(pdf_path: Path) -> str:
    """
    Parse the PDF and extract text content with improved handling for complex layouts.

    Parameters
    ----------
    pdf_path : Path
        The path to the PDF file.

    Returns
    -------
    str
        The text content of the PDF file as a single formatted string.
    """
    text_content = []

    try:
        with fitz.open(pdf_path) as pdf:
            for page_num in range(pdf.page_count):
                page = pdf[page_num]

                page_text = ""
                try:
                    page_text = page.get_text("blocks")
                    page_text = sorted(
                        page_text, key=lambda block: (block[1], block[0])
                    )
                    page_text = "\n".join(
                        block[4] for block in page_text if block[4].strip()
                    )
                except Exception as e:  # pylint: disable=broad-except
                    logging.warning(
                        f"Page {page_num + 1} could not be fully parsed: {e}"
                    )

                if page_text.strip():
                    text_content.append(page_text.strip())
                else:
                    logging.info(
                        f"Page {page_num + 1} contains no text or is unreadable."
                    )

    except Exception as e:  # pylint: disable=broad-except
        logging.error(f"Error parsing PDF: {e}")
        return ""

    formatted_text_content = "\n\n".join(text_content)

    return formatted_text_content


def create_pdf_partitions(
    pdf_text: str,
    partition_len: int,
    overlap: int = 0,
) -> List[str]:
    """
    Create partitions of the PDF text based on the specified length.

    Parameters
    ----------
    pdf_text : str
        The text content of the PDF file.
    partition_len : int
        The maximum length of each partition.
    overlap : int, optional
        The overlap between partitions, by default 0.

    Returns
    -------
    List[str]
        A list of text partitions.
    """

    partitions = []

    for i in range(0, len(pdf_text), partition_len):
        partition = pdf_text[max(0, i - overlap) : i + partition_len]
        partitions.append(partition)

    return partitions


def find_relevant_pdf_parts(
    pdf_path: Path,
    message: str,
    k: int,
) -> str:
    """
    Find the relevant parts of the PDF based on the message.

    Parameters
    ----------
    pdf_path : Path
        The path to the PDF file.
    message : str
        The message to search for in the PDF.
    k : int
        The number of relevant partitions to return.

    Returns
    -------
    str
        The relevant parts of the PDF based on the message.
    """

    def update_best_partitions(
        best_partitions: List[Tuple[str, float]],
        partition: str,
        distance: float,
    ) -> List[Tuple[str, float]]:
        if len(best_partitions) < k:
            best_partitions.append((partition, distance))
            best_partitions.sort(key=lambda x: x[1])
            return best_partitions

        for i, (_, dist) in enumerate(best_partitions):
            if distance < dist:
                best_partitions.insert(i, (partition, distance))
                best_partitions.pop()

        return best_partitions

    pdf_text = parse_pdf(pdf_path)
    partitions = create_pdf_partitions(pdf_text, 1024, 256)

    message_embedding = generate_embedding(message, normalize=True)

    best_partitions = []

    for partition in partitions:
        part_embedding = generate_embedding(partition, normalize=True)
        dist = find_distance_L2(message_embedding, part_embedding)
        best_partitions = update_best_partitions(best_partitions, partition, dist)

    return "\n\n".join(part for part, _ in best_partitions)


def ask_pdf(paper: ArxivPaper, message: str) -> str:
    """
    Ask a question to a research paper based on its content.

    Parameters
    ----------
    paper : ArxivPaper
        The paper to chat with.
    message : str
        The message to send to the paper.

    Returns
    -------
    str
        The response from the paper.
    """

    pdf_path = _download_paper_pdf(paper)
    relevant_parts = find_relevant_pdf_parts(pdf_path, message, 3)

    message_with_info = (
        f"# Question\n{message}\n\n# Supplemental Information\n{relevant_parts}"
    )

    client = Groq()

    completion = client.chat.completions.create(
        model="llama-3.2-3b-preview",
        messages=[
            {
                "role": "system",
                "content": QA_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": message_with_info,
            },
        ],
        temperature=0,
        max_tokens=2048,
        top_p=1,
        stream=False,
        stop=None,
    )

    result = completion.choices[0].message.content

    return result
