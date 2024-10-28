import numpy as np
from transformers import AutoModel, AutoTokenizer

SCIBERT_TOKENIZER = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
SCIBERT_MODEL = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")


def normalize_vector_L2(vector: np.ndarray) -> np.ndarray:
    """
    Normalize the input vector using L2 normalization.

    Parameters
    ----------
    vector : np.ndarray
        The input vector to normalize.

    Returns
    -------
    np.ndarray
        The normalized vector.
    """

    norm = np.linalg.norm(vector, ord=2)
    normalized_vector = vector / norm

    return normalized_vector


def find_distance_L2(
    vector1: np.ndarray,
    vector2: np.ndarray,
) -> float:
    """
    Find the L2 distance between two input vectors.

    Parameters
    ----------
    vector1 : np.ndarray
        The first input vector.
    vector2 : np.ndarray
        The second input vector.

    Returns
    -------
    float
        The L2 distance between the two input vectors.
    """

    distance = np.linalg.norm(vector1 - vector2, ord=2)

    return distance


def get_scibert_tokenizer() -> AutoTokenizer:
    """
    Returns the SciBERT tokenizer from the Hugging Face Transformers library.

    Parameters
    ----------
    None

    Returns
    -------
    AutoTokenizer
        The SciBERT tokenizer.
    """

    return SCIBERT_TOKENIZER


def get_scibert_model() -> AutoModel:
    """
    Returns the SciBERT model from the Hugging Face Transformers library.

    Parameters
    ----------
    None

    Returns
    -------
    AutoModel
        The SciBERT model.
    """

    return SCIBERT_MODEL


def generate_embedding(
    text: str,
    normalize: bool = False,
) -> np.ndarray:
    """
    Generate the embedding of the given text using the SciBERT model.

    Parameters
    ----------
    text : str
        The text to generate the embedding for.
    normalize : bool, optional
        Whether to normalize the embedding, by default False.

    Returns
    -------
    np.ndarray
        The embedding of the text.
    """

    model = get_scibert_model()
    tokenizer = get_scibert_tokenizer()

    inputs = tokenizer(text.lower(), return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()

    if normalize:
        embeddings = normalize_vector_L2(embeddings)

    return embeddings


def generate_embeddings(
    texts: list[str],
    normalize: bool = False,
) -> dict[str, np.ndarray]:
    """
    Generate the embeddings of the given texts using the SciBERT model.

    Parameters
    ----------
    texts : list[str]
        The list of texts to generate the embeddings for.
    normalize : bool, optional
        Whether to normalize the embeddings, by default False.

    Returns
    -------
    dict[str, np.ndarray]
        A dictionary containing the embeddings of the texts.
    """

    model = get_scibert_model()
    tokenizer = get_scibert_tokenizer()

    embeddings = {}

    for text in texts:
        inputs = tokenizer(
            text.lower(), return_tensors="pt", padding=True, truncation=True
        )
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()

        if normalize:
            embedding = normalize_vector_L2(embedding)

        embeddings[text] = embedding

    return embeddings
