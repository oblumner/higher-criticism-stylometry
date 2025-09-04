import pandas as pd
import spacy
from functools import lru_cache
from spacy.util import compile_infix_regex
from typing import List, Dict, Set, Optional

@lru_cache(maxsize=2)
def load_custom_nlp(model_name: str) -> spacy.language.Language:
    """
    Loads a spaCy model, customizes its tokenizer to reduce splitting on
    hyphens, and caches the result.
    Args: model_name (str): Name of the spaCy moddel to load (e.g., "en_core_web_sm").
    Returns: spacy.language.Language: The loaded and customized nlp object.
    """
    
    try:
        nlp = spacy.load(model_name)
        default_infixes = list(nlp.Defaults.infixes)
        custom_infixes = [pattern for pattern in default_infixes if "-" not in pattern]

        if custom_infixes != default_infixes:
            infix_re = compile_infix_regex(custom_infixes)
            nlp.tokenizer.infix_finditer = infix_re.finditer
        
        return nlp
    except OSError as e:
        error_message = (
            f"Error loading spaCy model '{model_name}'."
            f"Please make sure it's downloaded (e.g., run 'python -m spacy download {model_name}')"
            f"Original error: {e}"
        )
        print(error_message)
        raise IOError(error_message) from e
    except Exception as e:
        error_message = f"An unexpected error occurred while loading spaCy model '{model_name}': {e}"
        print(error_message)
        raise


def count_word_appearances(group_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Counts the number of DataFrames (experiments/iterations) each word appears in.
    Args: group_dfs (List[pd.DataFrame]): A list of pandas DataFrames.  Each DataFrame
    is expected to have a 'word' column containing the discriminating words from one 
    experiment/bootstrap iteration.
    Returns: pd.DataFrame: A DataFrame with 'word' and 'count' columns, sorted by count 
    in descending order. 
    """
    word_count: Dict[str, int] = {}

    for df in group_dfs:
        if df is not None and not df.empty and 'word' in df.columns:
            words_in_df: Set[str] = set(df['word'])
            for word in words_in_df:
                word_count[word] = word_count.get(word, 0) + 1
        else:
            print('Warning: Encountered an empty or invalid DataFrame in count_word_appearances.')
            pass

    if not word_count:
        return pd.DataFrame(columns=['word', 'count'])

    data = {
        'word': list(word_count.keys()),
        'count': list(word_count.values())
    }

    count_df = pd.DataFrame(data)
    count_df.sort_values(by='count', ascending=False, inplace=True)
    
    return count_df



