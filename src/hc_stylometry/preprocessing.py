import re
import spacy
from typing import List

def filter_and_clean_text(text: str, nlp: spacy.language.Language, pos_tags: List[str]) -> str:
    """
    Filters text based on POS tags, removes entities, and cleans the text.
    Args:
        text (str): The input text string.
        nlp (spacy.language.Language): The loaded spaCy NLP object.
        pos_tags (List[str]): A list of POS tags to keep (e.g., ['NOUN', 'VERB']).
    Returns:
        str: The cleaned and filtered text as a single string.
    """
    if not text.strip():
        return ""

    doc = nlp(text)
    
    # Regex pattern to exclude words containing both letters and digits
    letter_digit_pattern = re.compile(r'[a-zA-Z]+\d+|\d+[a-zA-Z]+')
    
    filtered_words = [
        token.text
        for token in doc
        if token.pos_ in pos_tags
        and token.ent_type_ == ""
        and not token.is_punct
        and not token.is_space
        and not letter_digit_pattern.search(token.text)  # Exclude words with letters + digits
    ]

    if not filtered_words:
        return ""
    
    filtered_text = " ".join(filtered_words)
    filtered_text = filtered_text.lower()
    filtered_text = re.sub(r'[^\w\s-]', '', filtered_text)
    filtered_text = re.sub(r'\s-\s|^-\s|\s-$|^-|-$', ' ', filtered_text)
    filtered_text = re.sub(r'\s+', ' ', filtered_text).strip()
    
    return filtered_text