import numpy as np
import pandas as pd
from sklearn.utils import resample
from typing import List, Tuple, Dict
from .core import higher_criticism
from .utils import count_word_appearances

def bootstrap_docs(
    model_1_cleaned_docs: List[str], 
    col_1_name: str, 
    model_2_cleaned_docs: List[str], 
    col_2_name: str, 
    n_iterations: int, 
    seeds_lists: List[List[int]]
) -> Tuple[List[pd.DataFrame], pd.DataFrame, List[float]]:
    
    """
    Performs bootstrapping for uncoupled lists of documents.
    Args:
        model_1_cleaned_docs: List of cleaned documents for corpus 1.
        col_1_name: Name for corpus 1.
        model_2_cleaned_docs: List of cleaned documents for corpus 2.  
        col_2_name: Name for corpus 2.  
        n_iterations: Number of bootstrapping iterations.
        seeds_lists: List of seed pairs for reproducibility.

    Returns:
        - List of DataFrames, each from one HC run on a bootstrap sample.
        - DataFrame of word counts across all bootstrap iterations. 
        - List of HC scores from each bootstrap iteration. 
    """

    hc_results_list: List[pd.DataFrame] = []
    hc_scores_list: List[float] = []

    if not model_1_cleaned_docs or not model_2_cleaned_docs:
        print("Warning: One or both document lists are empty in bootstrap_docs.")
        empty_count_df = pd.DataFrame(columns=['word', 'count'])
        return [], empty_count_df, []

    for i in range(n_iterations):
        seeds = seeds_lists[i]

        sample1_docs = resample(
            model_1_cleaned_docs,
            n_samples=len(model_1_cleaned_docs),
            replace=True,
            random_state=seeds[0]
        )

        sample1_text = '\n\n'.join(doc for doc in sample1_docs if doc)
        
        sample2_docs = resample(
            model_2_cleaned_docs,
            n_samples=len(model_2_cleaned_docs),
            replace=True,
            random_state=seeds[1]
        )

        sample2_text = '\n\n'.join(doc for doc in sample2_docs if doc)

        if not sample1_text.strip() or not sample2_text.strip():
            continue

        hc_score, hc_df = higher_criticism(sample1_text, col_1_name, sample2_text, col_2_name)
        hc_results_list.append(hc_df)
        hc_scores_list.append(hc_score)
    
    count_df = count_word_appearances(hc_results_list)

    return hc_results_list, count_df, hc_scores_list


def paired_bootstrap_docs(
    model_1_cleaned_docs: List[str], 
    col_1_name: str,
    model_2_cleaned_docs: List[str], 
    col_2_name: str, 
    n_iterations: int, 
    seeds_lists: List[List[int]]
) -> Tuple[List[pd.DataFrame], pd.DataFrame, List[float]]:

    """
    Performs bootstrapping for couple (paired) lists of documents.
    Assumes model_1_cleaned_docs[i] is paired with model_2_cleaned_docs[i].

    Args:
        model_1_cleaned_docs: List of cleaned documents for corpus 1.
        col_1_name: Name for corpus 1.
        model_2_cleaned_docs: List of cleaned documents for corpus 2. 
        col_2_name: Name for corpus 2.
        n_iterations: Number of bootstrapping iterations.
        seeds_lists: List of seed pairs for reproducibility. 
    
    Returns:
        - List of DataFrames, each from one HC run on a bootstrap sample.
        - DataFrame of word counts across all bootstrap iterations.
        - List of HC scores from each bootstrap iteration. 
    """

    hc_results_list: List[pd.DataFrame] = []
    hc_scores_list: List[float] = []

    if len(model_1_cleaned_docs) != len(model_2_cleaned_docs):
        raise ValueError('For paired bootstrapping, input document lists must have the same length.')
    
    if not model_1_cleaned_docs:
        print('Warning: Document lists are empty in paired_bootstrap_docs.')
        empty_count_df = pd.DataFrame(columns=['word', 'count'])
        return [], empty_count_df, []

    num_pairs = len(model_1_cleaned_docs)
    indices = list(range(num_pairs))

    for i in range(n_iterations):

        seeds = seeds_lists[i]
        resampled_indices = resample(
            indices,
            n_samples=num_pairs,
            replace=True,
            random_state=seeds[0]
        )

        sample1_docs = [model_1_cleaned_docs[idx] for idx in resampled_indices]
        sample2_docs = [model_2_cleaned_docs[idx] for idx in resampled_indices]

        sample1_text = '\n\n'.join(doc for doc in sample1_docs if doc)
        sample2_text = '\n\n'.join(doc for doc in sample2_docs if doc)

        if not sample1_text.strip() or not sample2_text.strip():
            continue

        hc_score, hc_df = higher_criticism(sample1_text, col_1_name, sample2_text, col_2_name)
        hc_results_list.append(hc_df)
        hc_scores_list.append(hc_score)

    count_df = count_word_appearances(hc_results_list)

    return hc_results_list, count_df, hc_scores_list