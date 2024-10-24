from transformers import AutoModel, AutoTokenizer
import numpy as np

TOKENIZER = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
MODEL = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")


def get_tokenizer() -> AutoTokenizer:
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

    return TOKENIZER


def get_model() -> AutoModel:
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

    return MODEL


def generate_embedding(text: str) -> np.ndarray:
    """
    Generate the embedding of the given text using the SciBERT model.

    Parameters
    ----------
    text : str
        The text to generate the embedding for.

    Returns
    -------
    np.ndarray
        The embedding of the text.
    """

    model = get_model()
    tokenizer = get_tokenizer()

    inputs = tokenizer(text.lower(), return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()

    return embeddings


def generate_embeddings(texts: list[str]) -> dict[str, np.ndarray]:
    """
    Generate the embeddings of the given texts using the SciBERT model.

    Parameters
    ----------
    texts : list[str]
        The list of texts to generate the embeddings for.

    Returns
    -------
    dict[str, np.ndarray]
        A dictionary containing the embeddings of the texts.
    """

    model = get_model()
    tokenizer = get_tokenizer()

    embeddings = {}

    for text in texts:
        inputs = tokenizer(
            text.lower(), return_tensors="pt", padding=True, truncation=True
        )
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        embeddings[text] = embedding

    return embeddings
