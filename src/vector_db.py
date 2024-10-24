import faiss
import numpy as np
import sqlite3
from pathlib import Path
import json
from arxiv_paper import ArxivPaper
from typing import List, Optional
from embedding_generator import generate_embeddings
from categorizer import MAX_SUB_CATEGORIES

DEFAULT_DATA_DIR = Path("vector_db")
DEFAULT_INDEX_PATH = DEFAULT_DATA_DIR / "index.faiss"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "arxiv_papers.db"

DEFAULT_DATA_DIR.mkdir(exist_ok=True, parents=True)


class VectorDB:
    def __init__(
        self,
        dim: int,
        db_path: str = DEFAULT_DB_PATH,
        index_path: Path = DEFAULT_INDEX_PATH,
    ) -> None:
        """
        Initialize the vector database with an SQLite database for metadata and FAISS for embeddings.

        Parameters
        ----------
        dim : int
            The dimension of the vectors.
        db_path : str, optional
            Path to the SQLite database.
        index_path : Path, optional
            Path to save the FAISS index file.
        """

        self.dim = dim
        self.index = faiss.index_factory(dim, "IDMap,Flat", faiss.METRIC_L2)
        self.index_path = index_path
        self.db_path = db_path

        # Create or connect to the SQLite database for storing metadata
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

        # Load the FAISS index if it exists
        self.load_index(path=index_path)

    def _create_tables(self) -> None:
        """
        Create the SQLite table for storing ArxivPaper metadata if it doesn't exist.
        """

        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                authors TEXT,
                url TEXT,
                summary TEXT,
                category TEXT,
                sub_categories TEXT
            )
        """
        )
        self.conn.commit()

    def add_data(self, papers: List[ArxivPaper], autosave: bool = True) -> None:
        """
        Add papers and embeddings to the vector database and store metadata in SQLite.

        Parameters
        ----------
        papers : List[ArxivPaper]
            The list of arXiv papers (metadata) to add.
        autosave : bool, optional
            Whether to save the index and metadata after adding.
        """

        cursor = self.conn.cursor()
        for idx, paper in enumerate(papers):
            # Check if paper's URL already exists in the database
            cursor.execute("SELECT id FROM papers WHERE url=?", (paper.url,))
            row = cursor.fetchone()

            if row is not None:
                continue

            # Insert the paper's metadata into the SQLite database
            cursor.execute(
                """
                INSERT INTO papers (title, authors, url, summary, category, sub_categories)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    paper.title,
                    ",".join(
                        paper.authors
                    ),  # Store authors as a comma-separated string
                    paper.url,
                    paper.summary,
                    paper.category,
                    ",".join(
                        paper.sub_categories
                    ),  # Store sub-categories as a comma-separated string
                ),
            )

            # Get the SQL ID of the inserted paper
            paper_id = cursor.lastrowid

            category_embeddings = np.asarray(
                list(generate_embeddings([paper.category]).values())
            ).reshape(-1, 768)
            sub_category_embeddings = np.asarray(
                list(generate_embeddings(paper.sub_categories).values())
            ).reshape(-1, 768)
            n_embeddings = len(category_embeddings) + len(sub_category_embeddings)

            embeddings = np.concatenate([category_embeddings, sub_category_embeddings])
            faiss.normalize_L2(embeddings)

            self.index.add_with_ids(
                # n_embeddings,
                embeddings,
                np.arange(
                    start=paper_id * (MAX_SUB_CATEGORIES + 1),
                    stop=(paper_id + 1) * (MAX_SUB_CATEGORIES + 1),
                )[:(n_embeddings)],
            )

        # Commit the changes to the SQLite database
        self.conn.commit()

        # Automatically save the index if autosave is True
        if autosave:
            self.save_index(path=self.index_path)

    def load_index(self, path: str) -> None:
        """
        Load the FAISS index if the files exist.

        Parameters
        ----------
        path : str
            Path to the FAISS index file.
        """

        # Load the FAISS index from the file if it exists
        if Path(path).exists():
            self.index = faiss.read_index(str(path))

    def save_index(self, path: str) -> None:
        """
        Save the FAISS index to disk.

        Parameters
        ----------
        path : str
            Path to save the FAISS index file.
        """

        faiss.write_index(self.index, str(path))

    def query(
        self,
        query_vector: np.ndarray,
        k: int,
        search_fields: Optional[List[str]] = None,
    ) -> List[ArxivPaper]:
        """
        Query the vector database for the nearest neighbors, with optional filtering based on category, sub-categories, and title.

        Parameters
        ----------
        query_vector : np.ndarray
            The query vector (embedding) for nearest neighbor search.
        k : int
            The number of nearest neighbors to retrieve.
        search_fields : Optional[List[str]]
            Fields to search in ('title', 'category', 'sub_categories'). Defaults to all fields.

        Returns
        -------
        List[ArxivPaper]
            The metadata of the most similar papers.
        """

        search_fields = search_fields or ["title", "category", "sub_categories"]

        faiss.normalize_L2(query_vector)

        # Perform FAISS nearest neighbors search
        distances, indices = self.index.search(query_vector.reshape(1, -1), k)

        paper_ids = np.unique((indices[0] / (MAX_SUB_CATEGORIES + 1)).astype(int)) + 1
        cursor = self.conn.cursor()

        # Construct the SQL query with optional filters
        query = "SELECT * FROM papers WHERE id IN ({})".format(
            ",".join("?" * len(paper_ids))
        )
        params = [int(x) for x in list(paper_ids)]

        # Execute the SQL query and fetch the results
        cursor.execute(query, params)
        rows = cursor.fetchall()

        # Convert the SQL results into ArxivPaper objects
        results = [
            ArxivPaper(
                title=row[1],
                authors=row[2].split(","),
                url=row[3],
                summary=row[4],
                category=row[5],
                sub_categories=row[6].split(","),
            )
            for row in rows
        ]

        return results
