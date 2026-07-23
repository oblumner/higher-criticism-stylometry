import random
from typing import List, Tuple
from sklearn.utils import resample


def split_half_null(
    cleaned_docs: List[str],
    seed: int,
) -> Tuple[List[str], List[str]]:
    """
    Constructs a null corpus pair by splitting a single corpus in half.

    Args:
        cleaned_docs: List of cleaned documents from one input corpus.
        seed: Random seed for shuffling reproducibility.

    Returns:
        (null_half_1, null_half_2): two non-overlapping document lists from the same source.
    """
    docs = list(cleaned_docs)
    rng = random.Random(seed)
    rng.shuffle(docs)
    mid = len(docs) // 2
    return docs[:mid], docs[mid:]


def resample_null(
    cleaned_docs: List[str],
    seed_1: int,
    seed_2: int,
) -> Tuple[List[str], List[str]]:
    """
    - Constructs a null corpus pair by resampling documents with replacement.
    - Keeps the null comparison size-matched to the real test comparison.
   
    Args:
        cleaned_docs: List of cleaned documents from one input corpus.
        seed_1: Random seed for the first resampled corpus.
        seed_2: Random seed for the second resampled corpus.

    Returns:
        (null_resample_1, null_resample_2): two document lists of the same size as the original corpus,
        each drawn with replacement from cleaned_docs.
    """
    n = len(cleaned_docs)
    null_1 = resample(cleaned_docs, n_samples=n, replace=True, random_state=seed_1)
    null_2 = resample(cleaned_docs, n_samples=n, replace=True, random_state=seed_2)
    return list(null_1), list(null_2)