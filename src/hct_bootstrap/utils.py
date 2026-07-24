import spacy
from functools import lru_cache
from spacy.util import compile_infix_regex

@lru_cache(maxsize=2)
def load_custom_nlp(model_name: str) -> spacy.language.Language:
    """
    Loads a spaCy model, customizes its tokenizer to reduce splitting on
    hyphens, and caches the result.
    Args: model_name (str): Name of the spaCy model to load (e.g., "en_core_web_sm").
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
            f"Error loading spaCy model '{model_name}'. "
            f"Please make sure it's downloaded (e.g., run 'python -m spacy download {model_name}') "
            f"Original error: {e}"
        )
        print(error_message)
        raise OSError(error_message) from e
    except Exception as e:
        error_message = f"An unexpected error occurred while loading spaCy model '{model_name}': {e}"
        print(error_message)
        raise


