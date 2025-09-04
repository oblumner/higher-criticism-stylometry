import numpy as np
import pandas as pd
from collections import Counter 
from typing import List, Dict, Tuple
from .TwoSampleHC.TwoSampleHC import two_sample_pvals, HC


def higher_criticism(
    text1: str, 
    corpus1_name: str, 
    text2: str, 
    corpus2_name: str
) -> Tuple[float, pd.DataFrame]:
    """
    Performs Higher Criticism analysis on two concatenated text corpora.
    Args: 
        text1 (str): The first corpus as a single string.
        corpus1_name (str): Name of first corpus.
        text2 (str): The second corpus as a single string.
        corpus2_name: Name of second corpus.
    Returns:
        Tuple[float, pd.DataFrame]: The HC score and a DataFrame of discriminating words.
    """
    if not text1.strip() or not text2.strip():
        print("Warning: One or both input texts to higher_criticism are empty.")
        return 0.0, pd.DataFrame(columns=['word', 'p-value', f"{corpus1_name}_frequency (%)", f"{corpus2_name}_frequency (%)", 'more_frequent_in'])

    
    text1_split = text1.split()
    text2_split = text2.split()

    if not text1_split or not text2_split:
        print("Warning: One or both input texts became empty after splitting.")
        return 0.0, pd.DataFrame(columns=['word', 'p-value', f"{corpus1_name}_frequency (%)", f"{corpus2_name}_frequency (%)", 'more_frequent_in'])   

    text1_freq = Counter(text1_split)
    text2_freq = Counter(text2_split)

    all_words = set(text1_freq.keys()) | set(text2_freq.keys())

    all_word_counts: Dict[str, List[int]] = {}
    for word in all_words:
        all_word_counts[word] = [text1_freq.get(word, 0), text2_freq.get(word, 0)]

    sorted_words = sorted(list(all_words))
    list_text1 = np.array([all_word_counts[word][0] for word in sorted_words])
    list_text2 = np.array([all_word_counts[word][1] for word in sorted_words])

    pvals = two_sample_pvals(list_text1, list_text2)
    hctest = HC(pvals)
    hc_score, pstar = hctest.HC()

    significant_words_data = []

    for i, word in enumerate(sorted_words):
        if pvals[i] < pstar:
            freq1_percent = (text1_freq.get(word, 0) / len(text1_split)) * 100 if len(text1_split) > 0 else 0
            freq2_percent = (text2_freq.get(word, 0) / len(text2_split)) * 100 if len(text2_split) > 0 else 0

            more_freq = 'tie'
            if freq1_percent > freq2_percent:
                more_freq = corpus1_name
            elif freq2_percent > freq1_percent:
                more_freq = corpus2_name
            
            significant_words_data.append({
                'word': word,
                'p_value': pvals[i],
                f"{corpus1_name}_frequency (%)": freq1_percent,
                f"{corpus2_name}_frequency (%)": freq2_percent,
                'more_frequent_in': more_freq
            })

    df = pd.DataFrame(significant_words_data)
    if not df.empty:
        df.sort_values(by='p_value', ascending=True, inplace=True)
    else:
        df = pd.DataFrame(columns=['word', 'p_value', f"{corpus1_name}_frequency (%)", f"{corpus2_name}_frequency (%)", 'more_frequent_in'])

    return hc_score, df