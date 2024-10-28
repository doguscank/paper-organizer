import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple, Union

import faiss
import numpy as np

from arxiv_paper import ArxivPaper
from categorizer import MAX_SUB_CATEGORIES
from embedding_generator import generate_embeddings

DEFAULT_DATA_DIR = Path("vector_db")
DEFAULT_INDEX_PATH = DEFAULT_DATA_DIR / "index.faiss"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "arxiv_papers.db"

DEFAULT_DATA_DIR.mkdir(exist_ok=True, parents=True)


class VectorDB:
    def __init__(
        self,
        dim: int,
        conn: Optional[sqlite3.Connection] = None,
        db_path: str = DEFAULT_DB_PATH,
        index_path: Path = DEFAULT_INDEX_PATH,
    ) -> None:
        """
        Initialize the vector database with an SQLite database for metadata and FAISS for embeddings.

        Parameters
        ----------
        dim : int
            The dimension of the vectors.
        conn : Optional[sqlite3.Connection], optional
            An existing SQLite connection to use.
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
        self.conn = conn or sqlite3.connect("arxiv_papers.db", check_same_thread=False)
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
                sub_categories TEXT,
                is_favorite INTEGER DEFAULT 0
            )
            """
        )

        self.conn.commit()

    def check_if_paper_exists(self, paper: ArxivPaper) -> bool:
        """
        Check if a paper already exists in the database based on its URL.

        Parameters
        ----------
        paper : ArxivPaper
            The arXiv paper to check.

        Returns
        -------
        bool
            True if the paper exists, False otherwise.
        """

        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM papers WHERE url=?", (paper.url,))
        row = cursor.fetchone()

        return row is not None

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
        for paper in papers:
            if self.check_if_paper_exists(paper):
                continue

            # Insert the paper's metadata into the SQLite database
            cursor.execute(
                """
                INSERT INTO papers (title, authors, url, summary, category, sub_categories, is_favorite)
                VALUES (?, ?, ?, ?, ?, ?, ?)
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
                    paper.is_favorite,
                ),
            )

            # Get the SQL ID of the inserted paper
            paper_id = cursor.lastrowid

            category_embeddings = np.asarray(
                list(generate_embeddings([paper.category]).values())
            ).reshape(-1, 768)
            sub_category_embeddings = np.asarray(
                list(generate_embeddings(paper.sub_categories).values())
            ).reshape(len(paper.sub_categories), 768)
            n_embeddings = len(category_embeddings) + len(sub_category_embeddings)

            embeddings = np.concatenate([category_embeddings, sub_category_embeddings])
            faiss.normalize_L2(embeddings)

            self.index.add_with_ids(
                # n_embeddings,
                embeddings,
                np.arange(
                    start=paper_id * (MAX_SUB_CATEGORIES + 1),
                    stop=(paper_id + 1) * (MAX_SUB_CATEGORIES + 1),
                    step=1,
                    dtype=int,
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
        min_similarity: float = 0.75,
        return_similarity: bool = False,
    ) -> Union[List[ArxivPaper], Tuple[List[ArxivPaper], np.ndarray]]:
        """
        Query the vector database for the nearest neighbors, with optional filtering
        based on category, sub-categories, and title.

        Parameters
        ----------
        query_vector : np.ndarray
            The query vector (embedding) for nearest neighbor search.
        k : int
            The number of nearest neighbors to retrieve.
        search_fields : Optional[List[str]]
            Fields to search in ('title', 'category', 'sub_categories'). Defaults to
            all fields.
        min_similarity : float
            Minimum cosine similarity for filtering the results.

        Returns
        -------
        List[ArxivPaper]
            The metadata of the most similar papers.
        """

        search_fields = search_fields or ["title", "category", "sub_categories"]

        faiss.normalize_L2(query_vector)

        # Perform FAISS nearest neighbors search
        distances, indices = self.index.search(query_vector.reshape(1, -1), k)
        distances, indices = distances[0], indices[0]
        similarities = 1 / (1 + distances)

        filtered_similarities = similarities[similarities > min_similarity]
        filtered_indices = indices[similarities > min_similarity]

        paper_ids = np.unique((filtered_indices / (MAX_SUB_CATEGORIES + 1)).astype(int))
        cursor = self.conn.cursor()

        # Construct the SQL query with optional filters
        query = f"SELECT * FROM papers WHERE id IN ({','.join('?' * len(paper_ids))})"
        params = [int(x) for x in list(paper_ids)]

        # Execute the SQL query and fetch the results
        cursor.execute(query, params)
        rows = cursor.fetchall()

        # Convert the SQL results into ArxivPaper objects
        results = [
            ArxivPaper(
                id=row[0],
                title=row[1],
                authors=row[2].split(","),
                url=row[3],
                summary=row[4],
                category=row[5],
                sub_categories=row[6].split(","),
                is_favorite=row[7],
            )
            for row in rows
        ]

        if return_similarity:
            return results, filtered_similarities
        return results

    def get_paper_by_id(self, paper_id: int) -> ArxivPaper:
        """
        Get a paper by its ID.

        Parameters
        ----------
        paper_id : int
            The ID of the paper.

        Returns
        -------
        ArxivPaper
            The paper with the given ID.
        """

        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM papers WHERE id=?", (paper_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return ArxivPaper(
            id=row[0],
            title=row[1],
            authors=row[2].split(","),
            url=row[3],
            summary=row[4],
            category=row[5],
            sub_categories=row[6].split(","),
            is_favorite=row[7],
        )

    def get_all_papers(self) -> List[ArxivPaper]:
        """
        Get all papers stored in the database.

        Returns
        -------
        List[ArxivPaper]
            A list of all papers in the database.
        """

        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM papers")
        rows = cursor.fetchall()

        return [
            ArxivPaper(
                id=row[0],
                title=row[1],
                authors=row[2].split(","),
                url=row[3],
                summary=row[4],
                category=row[5],
                sub_categories=row[6].split(","),
                is_favorite=row[7],
            )
            for row in rows
        ]

    def get_paper_by_title(self, title: str) -> ArxivPaper:
        """
        Get a paper by its title.

        Parameters
        ----------
        title : str
            The title of the paper.

        Returns
        -------
        ArxivPaper
            The paper with the given title.
        """

        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM papers WHERE title=?", (title,))
        row = cursor.fetchone()

        if row is None:
            return None

        return ArxivPaper(
            id=row[0],
            title=row[1],
            authors=row[2].split(","),
            url=row[3],
            summary=row[4],
            category=row[5],
            sub_categories=row[6].split(","),
            is_favorite=row[7],
        )

    def get_all_titles(self) -> List[str]:
        """
        Get all unique titles in the database.

        Returns
        -------
        List[str]
            A list of all unique titles in the database.
        """

        cursor = self.conn.cursor()
        cursor.execute("SELECT title FROM papers")
        rows = cursor.fetchall()

        return [row[0] for row in rows]

    def get_all_categories(self) -> List[str]:
        """
        Get all unique categories in the database.

        Returns
        -------
        List[str]
            A list of all unique categories in the database.
        """

        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT category FROM papers")
        rows = cursor.fetchall()

        return list({row[0] for row in rows})

    def get_all_sub_categories(self) -> List[str]:
        """
        Get all unique sub-categories in the database.

        Returns
        -------
        List[str]
            A list of all unique sub-categories in the database.
        """

        cursor = self.conn.cursor()
        cursor.execute("SELECT sub_categories FROM papers")
        rows = cursor.fetchall()

        sub_categories = [row[0].split(",") for row in rows]
        return list(
            {
                sub_category
                for sub_categories in sub_categories
                for sub_category in sub_categories
            }
        )

    def get_papers_by_category(self, category: str) -> List[ArxivPaper]:
        """
        Get papers by category.

        Parameters
        ----------
        category : str
            The category of the papers.

        Returns
        -------
        List[ArxivPaper]
            The papers with the given category.
        """

        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM papers WHERE category=?", (category,))
        rows = cursor.fetchall()

        return [
            ArxivPaper(
                id=row[0],
                title=row[1],
                authors=row[2].split(","),
                url=row[3],
                summary=row[4],
                category=row[5],
                sub_categories=row[6].split(","),
                is_favorite=row[7],
            )
            for row in rows
        ]

    def get_papers_by_sub_category(self, sub_category: str) -> List[ArxivPaper]:
        """
        Get papers by sub-category.

        Parameters
        ----------
        sub_category : str
            The sub-category of the papers.

        Returns
        -------
        List[ArxivPaper]
            The papers with the given sub-category.
        """

        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM papers WHERE sub_categories LIKE ?",
            ("%" + sub_category + "%",),
        )
        rows = cursor.fetchall()

        return [
            ArxivPaper(
                id=row[0],
                title=row[1],
                authors=row[2].split(","),
                url=row[3],
                summary=row[4],
                category=row[5],
                sub_categories=row[6].split(","),
                is_favorite=row[7],
            )
            for row in rows
        ]

    def get_favorite_papers(self) -> List[ArxivPaper]:
        """
        Get all favorite papers.

        Returns
        -------
        List[ArxivPaper]
            A list of all favorite papers in the database.
        """

        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM papers WHERE is_favorite = 1")
        rows = cursor.fetchall()
        papers = []
        for row in rows:
            paper = ArxivPaper(
                id=row[0],
                title=row[1],
                authors=row[2].split(","),
                url=row[3],
                summary=row[4],
                category=row[5],
                sub_categories=row[6].split(","),
                is_favorite=bool(row[7]),
            )
            papers.append(paper)
        return papers

    def add_favorite(self, paper_id: int) -> None:
        """
        Add a paper to the favorites.

        Parameters
        ----------
        paper_id : int
            The ID of the paper to add to favorites.

        Returns
        -------
        None
        """

        cursor = self.conn.cursor()
        cursor.execute("UPDATE papers SET is_favorite = 1 WHERE id = ?", (paper_id,))
        self.conn.commit()

    def remove_favorite(self, paper_id: int) -> None:
        """
        Remove a paper from the favorites.

        Parameters
        ----------
        paper_id : int
            The ID of the paper to remove from favorites.

        Returns
        -------
        None
        """

        cursor = self.conn.cursor()
        cursor.execute("UPDATE papers SET is_favorite = 0 WHERE id = ?", (paper_id,))
        self.conn.commit()

    def is_favorite(self, paper_id: int) -> bool:
        """
        Check if a paper is in the favorites.

        Parameters
        ----------
        paper_id : int
            The ID of the paper to check.

        Returns
        -------
        bool
            True if the paper is in the favorites, False otherwise.
        """

        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT 1 FROM papers WHERE id = ? AND is_favorite = 1", (paper_id,)
        )
        return cursor.fetchone() is not None
