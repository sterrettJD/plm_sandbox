import numpy as np


KEYS = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "<PAD>", "<MASK>", "<BOS>", "<UNK>"]
IDS = list(range(len(KEYS)))
VOCAB = {key: id for key, id in zip(KEYS, IDS)}
REV_VOCAB = {id: key for key, id in zip(KEYS, IDS)}
MASK_TOKEN = VOCAB["<MASK>"]
PAD_TOKEN = VOCAB["<PAD>"]
BOS_TOKEN = VOCAB["<BOS>"]
UNK_TOKEN = VOCAB["<UNK>"]


def tokenize(sequence: str) -> np.ndarray:
    """
    Tokenizes a raw protein sequence string into a list of token IDs.

    Args:
        sequence (str): The raw protein sequence (e.g., "MVLSPADKTNV").

    Returns:
        list[int]: A list of token IDs corresponding to each character in the input sequence.
    """
    token_ids = [VOCAB.get(char, VOCAB["<UNK>"]) for char in sequence]
    
    return np.array(token_ids)


def de_tokenize(token_ids: np.ndarray) -> str:
    """
    Decodes a list of token IDs back into the original protein sequence string.

    Args:
        token_ids (list[int]): A list of token IDs produced by the tokenize function.

    Returns:
        str: The original protein sequence string, reconstructed from the token IDs.
    """
    
    # We'll map each token ID back to its corresponding character in our vocabulary dictionary.
    
    decoded = [REV_VOCAB.get(id) for id in token_ids]
    
    return "".join(decoded)

