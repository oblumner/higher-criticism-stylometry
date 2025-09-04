import random
import spacy
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Dict, Tuple, Optional, Any
from .TwoSampleHC.TwoSampleHC import two_sample_pvals, HC
from .preprocessing import filter_and_clean_text
from .sampling import bootstrap_docs, paired_bootstrap_docs
from .utils import load_custom_nlp, count_word_appearances
from .core import higher_criticism

_HAS_IPYTHON = False
try:
    from IPython.display import display, HTML
    _HAS_PYTHON = True
except ImportError:
    pass

_HAS_PLOTTING = False
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    _HAS_PLOTTING = True
except ImportError:
    pass

def discriminate(
    corpus1_docs: List[str], 
    corpus1_name: str, 
    corpus2_docs: List[str],
    corpus2_name: str,
    coupled: bool, 
    nlp: Optional[spacy.language.Language] = None, 
    pos_tags: List[str] = ['NOUN', 'PROPN', 'ADJ', 'VERB', 'ADV'],  
    n_iterations: int = 100,
    random_seed: int = 42,
    default_spacy_model: str = "en_core_web_sm"
) -> Dict[str, Any]:
    """
    Identifies stable discriminating words between two text corpora using Higher Criticism and bootstrapping.
    Args:
        corpus1_docs: A list of document strings for the first corpus.
        corpus1_name: Name for the first corpus (e.g., 'GPT-4o')
        corpus2_docs: A list of document strings for the second corpus.
        corpus2_name: Name for the second corpus (e.g., 'GPT-3.5-Turbo')
        coupled: True if corpus1_docs[i] is paired with corpus2_docs[i].
                 Requires lists to be of the same length if True.
        nlp: Pre-loaded spaCy model.  If None, loads default_spacy_model.
        pos_tags: POS tags to consider for analysis. 
        n_iterations: Number of bootstrap iterations.
        random_seed: Seed for reproducibility.
        default_spacy_model: default spaCy model to load. 
    
    Returns:
        A dictionary containing analysis results, including:
        - 'bootstrap_summary_df': DataFrame of words, their count and percentage of appearance
           in bootstrap iterations, and p-value from the full data run. 
        - 'bootstrap_hc_scores': List of HC scores from each bootstrap iteration.
        - 'full_data_hc_score': Overall HC score from a single run on all data.
        - 'full_data_results_df': DataFrame of discriminating words from the single run on all data.
    """
    active_nlp: spacy.language.Language
    if nlp is None:
        try: 
            active_nlp = load_custom_nlp(default_spacy_model)
        except (OSError, IOError, Exception) as e:
            raise RuntimeError(f"Failed to load/customize spaCy model '{default_spacy_model}'") from e
    else:
        active_nlp = nlp

    random.seed(random_seed)
    seeds_lists = [[random.randint(0, 2**32 - 1), random.randint(0, 2**32 - 1)] for _ in range(n_iterations)]

    #clean text

    model_1_list_clean = []
    model_2_list_clean = []

    if coupled:
        if len(corpus1_docs) != len(corpus2_docs):
            raise ValueError("For coupled analysis, initial input lists must have the same length.")
        
        for doc1, doc2 in zip(corpus1_docs, corpus2_docs):
            cleaned_doc1 = filter_and_clean_text(doc1, active_nlp, pos_tags)
            cleaned_doc2 = filter_and_clean_text(doc2, active_nlp, pos_tags)

            if cleaned_doc1.strip() and cleaned_doc2.strip():
                model_1_list_clean.append(cleaned_doc1)
                model_2_list_clean.append(cleaned_doc2)

    else:
        temp_list_1 = [filter_and_clean_text(doc, active_nlp, pos_tags) for doc in corpus1_docs]
        temp_list_2 = [filter_and_clean_text(doc, active_nlp, pos_tags) for doc in corpus2_docs]

        model_1_list_clean = [doc for doc in temp_list_1 if doc.strip()]
        model_2_list_clean = [doc for doc in temp_list_2 if doc.strip()]


    if not model_1_list_clean or not model_2_list_clean:
        return {
            "bootstrap_summary_df": pd.DataFrame(columns=['word', 'count', 'percent', 'p-value', 'more_frequent_in']),
            "bootstrap_hc_scores": [],
            "full_data_hc_score": 0.0,
            "full_data_results_df": pd.DataFrame()
        }

    text1 = '\n\n'.join(model_1_list_clean)
    text2 = '\n\n'.join(model_2_list_clean)
        
    #Higher Criticism over the 2 full corpora
    hc_full_score, hc_full_df = higher_criticism(text1, corpus1_name, text2, corpus2_name)
    
    #Bootstrapping Full Text
    if coupled:
        if len(model_1_list_clean) != len(model_2_list_clean):
            raise ValueError(
                f"For coupled analysis, corpus1 ({len(model_2_list_clean)} docs) and "
                f"corpus2 ({len(model_2_list_clean)} docs) must have the same number of documents after cleaning."
            )
        
        hc_bootstrap_runs_dfs, counts_bootstrap_df, bootstrap_hc_scores = paired_bootstrap_docs(model_1_list_clean, corpus1_name, model_2_list_clean, corpus2_name, n_iterations, seeds_lists)
    else:
        hc_bootstrap_runs_dfs, counts_bootstrap_df, bootstrap_hc_scores = bootstrap_docs(model_1_list_clean, corpus1_name, model_2_list_clean, corpus2_name, n_iterations, seeds_lists)
        
    if not counts_bootstrap_df.empty:
        counts_bootstrap_df['percent'] = (counts_bootstrap_df['count'] / n_iterations) * 100
        if not hc_full_df.empty and 'word' in hc_full_df.columns and 'p_value' in hc_full_df.columns:
            counts_bootstrap_df = counts_bootstrap_df.merge(
                hc_full_df[['word', 'p_value', 'more_frequent_in']], on='word', how='left'
            )
        else:
            counts_bootstrap_df['p_value'] = np.nan
            counts_bootstrap_df['more_frequent_in'] = None
        
        counts_bootstrap_df.sort_values(by=['count', 'p_value'], ascending=[False, True], inplace=True)
    else:
        counts_bootstrap_df = pd.DataFrame(columns=['word', 'count', 'percent', 'p_value', 'more_frequent_in'])

    results = {
        "bootstrap_summary_df": counts_bootstrap_df,
        "bootstrap_hc_scores": bootstrap_hc_scores,
        "full_data_hc_score": hc_full_score,
        "full_data_results_df": hc_full_df
    }
    return results, text1, text2


def analyze_and_display(
    corpus1_docs: List[str],
    corpus1_name: str,
    corpus2_docs: List[str],
    corpus2_name: str,
    coupled: bool,
    nlp: Optional[spacy.language.Language] = None,
    default_spacy_model: str = 'en_core_web_sm',
    pos_tags: List[str] = ['NOUN', 'PROPN', 'ADJ', 'VERB', 'ADV'],
    n_iterations: int = 100,
    random_seed: int = 42,
    top_n_full_results: int = 10
) -> Dict[str, Any]:
    """
    Performs Higher Criticism analysis and displays a summary report.  
    """
    print("=== Higher Criticism Analysis Report ===")
    print()
    print(f"Comparing: '{corpus1_name}' vs. '{corpus2_name}'")
    print(f"Coupled Analysis: {coupled}")
    print(f"Bootstrap Iterations: {n_iterations}")
    print("--------------------")

    try:
        results_dict, corpus1, corpus2 = discriminate(
            corpus1_docs=corpus1_docs,
            corpus1_name=corpus1_name, 
            corpus2_docs=corpus2_docs,
            corpus2_name=corpus2_name,
            coupled=coupled,
            nlp=nlp,
            default_spacy_model=default_spacy_model, 
            pos_tags=pos_tags,
            n_iterations=n_iterations,
            random_seed=random_seed
        )
    except RuntimeError as e:
        print(f"Analysis could not be completed: {e}")
        return {
            "error": str(e),
            "bootstrap_summary_df": pd.DataFrame(),
            "bootstrap_hc_scores": [],
            "full_data_hc_score": None, 
            "full_data_results_df": pd.DataFrame()
        }
    
    print(f"\n --- Corpus Information (Post-Cleaning) ---")
    print(f'# of words in {corpus1_name}: {len(corpus1.split())}')
    print(f'# of words in {corpus2_name}: {len(corpus2.split())}')

    print(f"\n--- Higher Criticism Results (Analysis on Full Text) ---")
    full_df = results_dict.get("full_data_results_df")
    if full_df is not None and not full_df.empty:
        display_df_subset = full_df.head(top_n_full_results)
        if _HAS_IPYTHON:
            display(HTML(display_df_subset.to_html(index=False)))
        else:
            print(display_df_subset.to_string(index=False))
        if len(full_df) > top_n_full_results:
            print(f"(Showing top {top_n_full_results} of {len(full_df)} discriminating words from full data run)")
    else:
        print(f"No discriminating words found.")

    full_hc_score = results_dict.get('full_data_hc_score')
    if full_hc_score is not None:
        print()
        print(f"Overall HC Score (Full Data): {full_hc_score:.4f}")
    
    print()
    print(f"--- Stable Words Found in 100% of Bootstrap Iterations ---")
    bootstrap_summary_df = results_dict.get("bootstrap_summary_df")
    if bootstrap_summary_df is not None and not bootstrap_summary_df.empty and 'count' in bootstrap_summary_df.columns:
        stable_words_df = bootstrap_summary_df[bootstrap_summary_df['count'] == n_iterations]
        if not stable_words_df.empty:
            words_for_corpus1 = stable_words_df[
                stable_words_df['more_frequent_in'] == corpus1_name
            ]['word'].tolist()

            words_for_corpus2 = stable_words_df[
                stable_words_df['more_frequent_in'] == corpus2_name
            ]['word'].tolist()

            print(f"\nWords more frequent in '{corpus1_name}':")
            if words_for_corpus1:
                print(", ".join(words_for_corpus1))
            else:
                print("None")
            
            print(f"\nWords more frequent in '{corpus2_name}':")
            if words_for_corpus2:
                print(", ".join(words_for_corpus2))
            else:
                print("None")
        else:
            print(f"No words survived all {n_iterations} bootstrap iterations.")
    else:
        print("Bootstrap summary not available to identify stable words.")


    print(f"\n--- Histogram of Bootstrap Word Count Frequencies ---")
    if bootstrap_summary_df is not None and not bootstrap_summary_df.empty and 'count' in bootstrap_summary_df.columns:
        if _HAS_PLOTTING:
            try:
                plt.figure(figsize=(10, 6))
                sns.histplot(bootstrap_summary_df['count'], discrete=True, kde=False)
                plt.title(f"Frequency of Word Stability across {n_iterations} Bootstrap Iterations")
                plt.xlabel("Number of Bootstrap Iterations a Word Appeared In")
                plt.ylabel("Number of Unique Words")
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Error plotting histogram: {e}")
        else:
            print("Plotting libraries (matplotlib, seaborn) not installed. Histogram not shown.")
            print("To enable plots, you might need to install them, e.g.:")
            print("   pip install hc_stylometry[display]")
            print("   or: pip install matplotlib seaborn")
            print("\nSummary of word counts from bootstrap:")
            print(bootstrap_summary_df['count'].value_counts().sort_index(ascending=False).to_string())
    else:
        print("Bootstrap summary 'count' data not available for histogram")

    
    bootstrap_hc_scores = results_dict.get("bootstrap_hc_scores")
    if bootstrap_hc_scores and len(bootstrap_hc_scores) > 0:
        avg_bootstrap_hc_score = np.mean(bootstrap_hc_scores)
        print(f"\nAverage HC Score over {n_iterations} Bootstrap Iterations: {avg_bootstrap_hc_score:.4f}")
    else:
        print("\nBootstrap HC scores not available or empty, cannot calculate average.")
    
    print("\n=== End of Report ===")
    return results_dict 

    



