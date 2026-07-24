import spacy
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union

from .preprocessing import filter_and_clean_text
from .utils import load_custom_nlp
from .core import higher_criticism
from .iterative_selection import discriminate_until_rule, plot_fdp_curve

_HAS_IPYTHON = False
try:
    from IPython.display import display, HTML
    _HAS_IPYTHON = True
except ImportError:
    pass

_HAS_PLOTTING = False
try:
    import matplotlib.pyplot as plt
    _HAS_PLOTTING = True
except ImportError:
    pass


def _clean_corpora(
    corpus1_docs: List[str],
    corpus2_docs: List[str],
    coupled: bool,
    active_nlp: spacy.language.Language,
    pos_tags: Optional[List[str]],
) -> Tuple[List[str], List[str]]:
    """
    Clean both corpora with the preprocessing pipeline.

    In the coupled case, a document pair is kept only if BOTH documents are
    non-empty after cleaning, so the prompt-level pairing is preserved.
    In the uncoupled case, each corpus is cleaned independently and empty
    documents are dropped.
    """
    model_1_clean: List[str] = []
    model_2_clean: List[str] = []

    if coupled:
        if len(corpus1_docs) != len(corpus2_docs):
            raise ValueError(
                "For coupled analysis, input lists must have the same length."
            )
        for doc1, doc2 in zip(corpus1_docs, corpus2_docs):
            c1 = filter_and_clean_text(doc1, active_nlp, pos_tags)
            c2 = filter_and_clean_text(doc2, active_nlp, pos_tags)
            if c1.strip() and c2.strip():
                model_1_clean.append(c1)
                model_2_clean.append(c2)
    else:
        tmp1 = [filter_and_clean_text(d, active_nlp, pos_tags) for d in corpus1_docs]
        tmp2 = [filter_and_clean_text(d, active_nlp, pos_tags) for d in corpus2_docs]
        model_1_clean = [d for d in tmp1 if d.strip()]
        model_2_clean = [d for d in tmp2 if d.strip()]

    return model_1_clean, model_2_clean


def discriminate(
    corpus1_docs: List[str],
    corpus1_name: str,
    corpus2_docs: List[str],
    corpus2_name: str,
    coupled: bool,
    q: Optional[float] = None,
    null_method: str = 'split',
    null_corpus1_docs: Optional[List[str]] = None,
    null_corpus2_docs: Optional[List[str]] = None,
    null_coupled: Optional[Union[bool, Tuple[bool, bool]]] = None,
    nlp: Optional[spacy.language.Language] = None,
    pos_tags: Optional[List[str]] = None,
    max_t: int = 1000,
    random_seed: int = 42,
    default_spacy_model: str = "en_core_web_sm",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Identify stable discriminating words between two text corpora using
    Higher Criticism with bootstrap stabilization and a within-corpus null.

    The number of bootstrap iterations is NOT fixed by the user. Instead, the
    procedure runs iterations until a data-driven stopping rule is satisfied:

        * q is None    -> STRICT rule: stop when no null word survives
                          (N0(n) == 0).
        * q is a float -> FDR rule:    stop when (1 + N0(n)) / max(N(n), 1) <= q.

    The null comparison is constructed via one of three methods:

        * null_method='split'    -> split each input corpus into two halves.
        * null_method='resample' -> draw two same-size with-replacement
                                     resamples from each input corpus.
        * null_method='user'     -> the user inputs a corresponding null for each input corpus

    
    The final selection is the set of test words that survived all iterations
    up to the stopping point, intersected with the single-run HCT candidate set
    (so BS-HCT is always a subset of HCT).

    Args:
        corpus1_docs, corpus2_docs: lists of document strings.
        corpus1_name, corpus2_name: corpus labels.
        coupled: True if corpus1_docs[i] is prompt-paired with corpus2_docs[i].
                 Affects only the TEST bootstrap. 
                 The 'split' / 'resample' nulls are always uncoupled; 
                 a user-supplied null follows null_coupled  
        q: None for the strict (no-null-survivors) rule, or a non-negative
           float for the FDR rule.
        null_method: 'split' / 'resample' / 'user'
            if null_method: 'user' is used, the following should be added:
                null_corpus1_docs: list of document strings
                null_corpus2_docs: list of document strings
                null_coupled: a single boolean if both null comparisons are either True / False, 
                              or a tuple of booleans if one is True and one is False. 
        nlp: pre-loaded spaCy model; if None, default_spacy_model is loaded.
        pos_tags: POS tags to retain, or None (default) to keep all POS.
        max_t: safety cap on the number of independent experiments (t).
        random_seed: seed for reproducibility.
        default_spacy_model: spaCy model name to load if nlp is None.
        verbose: print a per-iteration trace of N(n), N0(n), FDP+(n)

    Returns:
        dict with:
            selected_words      : sorted list of stable discriminating words
            selected_words_df   : those words with 'p_value' and 'more_frequent_in', ascending by p-value
            n_rule, t_stop      : iteration at which the rule fired (or None);
                                  both keys hold the same value
            fdp_at_stop         : FDP+(n) at the stopping iteration (or None)
            rule                : 'strict' or 'fdr'
            q, null_method      : echoed settings
            iterations_run      : number of iterations actually run
            empty_reason        : explanation if the selection is empty, else None
            fdp_curve           : list of (n, N(n), N0(n), FDP+(n)) tuples
            full_data_hc_score  : HC score on the full corpora
            full_data_results_df: HCT-selected words on the full corpora
            test_survivors_df   : surviving test words at the stop (['word'])
            null_survivors_dfs  : per-null surviving words (list of ['word'])
            cleaned_corpus1, cleaned_corpus2: cleaned text (joined), for info
    """
    # --- Load spaCy ---
    if nlp is None:
        try:
            active_nlp = load_custom_nlp(default_spacy_model)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load/customize spaCy model '{default_spacy_model}'"
            ) from e
    else:
        active_nlp = nlp

    # --- Clean ---
    model_1_clean, model_2_clean = _clean_corpora(
        corpus1_docs, corpus2_docs, coupled, active_nlp, pos_tags
    )

    if not model_1_clean or not model_2_clean:
        return {
            "selected_words": [],
            "n_rule": None,
            "t_stop": None,
            "fdp_at_stop": None,
            "fdp_curve": [],
            "selected_words_df": pd.DataFrame(columns=['word', 'p_value', 'more_frequent_in']),
            "rule": 'strict' if q is None else 'fdr',
            "q": q,
            "null_method": null_method,
            "iterations_run": 0,
            "empty_reason": "One or both corpora were empty after cleaning.",
            "full_data_hc_score": 0.0,
            "full_data_results_df": pd.DataFrame(),
            "test_survivors_df": pd.DataFrame(columns=['word']),
            "null_survivors_dfs": [],
            "cleaned_corpus1": "",
            "cleaned_corpus2": "",
        }

    # --- Full-corpus HCT (candidate set) ---
    text1 = '\n\n'.join(model_1_clean)
    text2 = '\n\n'.join(model_2_clean)
    hc_full_score, hc_full_df, _hc_all_df = higher_criticism(
        text1, corpus1_name, text2, corpus2_name
    )
    hct_words = set(hc_full_df['word']) if not hc_full_df.empty else set()

    # --- If user-supplied nulls: clean each (original vs user-null) pair ---
    user_null_pairs = None
    if null_method == 'user':
        if null_corpus1_docs is None or null_corpus2_docs is None:
            raise ValueError(
                "null_method='user' requires null_corpus1_docs and "
                "null_corpus2_docs (one user-supplied null corpus per input)."
            )
        # Normalize null_coupled to a (bool, bool) per-null tuple.
        if null_coupled is None:
            nc = (coupled, coupled)          # default: match the test coupling
        elif isinstance(null_coupled, (list, tuple)):
            if len(null_coupled) != 2:
                raise ValueError("null_coupled must be a bool or length-2 "
                                 "sequence (one flag per null comparison).")
            nc = (bool(null_coupled[0]), bool(null_coupled[1]))
        else:
            nc = (bool(null_coupled), bool(null_coupled))

        # Each null comparison is original-corpus vs its user null corpus,
        # cleaned together (coupled if the user generated matched prompts).
        nA_orig, nA_null = _clean_corpora(corpus1_docs, null_corpus1_docs,
                                          nc[0], active_nlp, pos_tags)
        nB_orig, nB_null = _clean_corpora(corpus2_docs, null_corpus2_docs,
                                          nc[1], active_nlp, pos_tags)
        if not nA_orig or not nA_null or not nB_orig or not nB_null:
            raise ValueError("A user null comparison was empty after cleaning.")
        user_null_pairs = [
            (nA_orig, nA_null, 'null_A', nc[0]),
            (nB_orig, nB_null, 'null_B', nc[1]),
        ]

    # --- Iterative null-calibrated bootstrap selection ---
    result = discriminate_until_rule(
        test_docs_a=model_1_clean, name_a=corpus1_name,
        test_docs_b=model_2_clean, name_b=corpus2_name,
        coupled=coupled,
        hct_words=hct_words,
        q=q,
        null_method=null_method,
        user_null_pairs=user_null_pairs,
        max_t=max_t,
        random_seed=random_seed,
        verbose=verbose
    )

    # --- Attach full-corpus HCT info and cleaned text ---
    result["full_data_hc_score"] = hc_full_score
    result["full_data_results_df"] = hc_full_df
    result["cleaned_corpus1"] = text1
    result["cleaned_corpus2"] = text2

    # --- Build a DataFrame of the selected words with their HCT p-value and
    #     which corpus each is more frequent in (joined from the full-corpus
    #     HCT run). Kept in ascending p-value order. ---
    selected = result.get("selected_words", [])
    if selected and hc_full_df is not None and not hc_full_df.empty:
        keep_cols = ['word', 'p_value', 'more_frequent_in']
        keep_cols = [c for c in keep_cols if c in hc_full_df.columns]
        sel_df = hc_full_df[hc_full_df['word'].isin(selected)][keep_cols].copy()
        if 'p_value' in sel_df.columns:
            sel_df.sort_values(by='p_value', ascending=True, inplace=True)
        sel_df.reset_index(drop=True, inplace=True)
    else:
        sel_df = pd.DataFrame(columns=['word', 'p_value', 'more_frequent_in'])
    result["selected_words_df"] = sel_df

    return result


def analyze_and_display(
    corpus1_docs: List[str],
    corpus1_name: str,
    corpus2_docs: List[str],
    corpus2_name: str,
    coupled: bool,
    q: Optional[float] = None,
    null_method: str = 'split',
    null_corpus1_docs: Optional[List[str]] = None,
    null_corpus2_docs: Optional[List[str]] = None,
    null_coupled: Optional[Union[bool, Tuple[bool, bool]]] = None,
    nlp: Optional[spacy.language.Language] = None,
    default_spacy_model: str = 'en_core_web_sm',
    pos_tags: Optional[List[str]] = None,
    max_t: int = 1000,
    random_seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run the null-calibrated BS-HCT analysis and print a summary report.
    """
    rule_desc = "strict (stop when no null word survives)" if q is None \
        else f"FDR (stop when FDP+ <= {q})"

    if null_method == 'user':
        null_desc = "user-supplied (corpus vs user null corpus)"
    else:
        null_desc = f"{null_method} (always uncoupled)"

    print("=== BS-HCT Analysis Report ===\n")
    print(f"Comparing: '{corpus1_name}' vs. '{corpus2_name}'")
    print(f"Coupled test comparison: {coupled}")
    print(f"Null construction: {null_desc}")
    print(f"Stopping rule: {rule_desc}")
    print(f"Max t: {max_t}")
    print("--------------------")

    try:
        res = discriminate(
            corpus1_docs=corpus1_docs, corpus1_name=corpus1_name,
            corpus2_docs=corpus2_docs, corpus2_name=corpus2_name,
            coupled=coupled, q=q, null_method=null_method,
            null_corpus1_docs=null_corpus1_docs,
            null_corpus2_docs=null_corpus2_docs,
            null_coupled=null_coupled,
            nlp=nlp, default_spacy_model=default_spacy_model,
            pos_tags=pos_tags, max_t=max_t,
            random_seed=random_seed,
            verbose=verbose
        )
    except RuntimeError as e:
        print(f"Analysis could not be completed: {e}")
        return {"error": str(e), "selected_words": [], "n_rule": None}

    print(f"\n--- Corpus Information (Post-Cleaning) ---")
    print(f"# words in {corpus1_name}: {len(res['cleaned_corpus1'].split())}")
    print(f"# words in {corpus2_name}: {len(res['cleaned_corpus2'].split())}")

    print(f"\n--- Full-Corpus HCT ---")
    full_df = res.get("full_data_results_df")
    n_hct = len(full_df) if full_df is not None else 0
    print(f"HCT candidate words: {n_hct}")
    score = res.get("full_data_hc_score")
    if score is not None:
        print(f"Full-corpus HC score: {score:.4f}")

    print(f"\n--- Stopping ---")
    if res["n_rule"] is not None:
        print(f"Rule fired at iteration n = {res['n_rule']} "
              f"(ran {res['iterations_run']} iterations).")
    else:
        print(f"Rule never fired within max_t={max_t}.")
    if res.get("empty_reason"):
        print(f"Note: {res['empty_reason']}")

    print(f"\n--- Stable Discriminating Words "
          f"({len(res['selected_words'])}) ---")
    selected = res["selected_words"]
    if selected:
        # Split by which corpus each word favors, using the full-corpus HCT df.
        favor = {}
        if full_df is not None and not full_df.empty \
                and 'more_frequent_in' in full_df.columns:
            favor = dict(zip(full_df['word'], full_df['more_frequent_in']))
        w1 = [w for w in selected if favor.get(w) == corpus1_name]
        w2 = [w for w in selected if favor.get(w) == corpus2_name]
        print(f"\nMore frequent in '{corpus1_name}':")
        print(", ".join(w1) if w1 else "None")
        print(f"\nMore frequent in '{corpus2_name}':")
        print(", ".join(w2) if w2 else "None")
        other = [w for w in selected if w not in set(w1) | set(w2)]
        if other:
            print(f"\n(Unattributed): {', '.join(other)}")

        # Full table with binomial-allocation p-values.
        sel_df = res.get("selected_words_df")
        if sel_df is not None and not sel_df.empty:
            print("\nSelected words with HCT p-values:")
            if _HAS_IPYTHON:
                display(HTML(sel_df.to_html(index=False)))
            else:
                print(sel_df.to_string(index=False))
    else:
        print("None.")

    print("\n=== End of Report ===")

    # --- FDP+ / survival-count curve over t ---
    if _HAS_PLOTTING:
        try:
            ax = plot_fdp_curve(res)
            if ax is not None:
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"Could not render FDP+ curve: {e}")
    else:
        print("\n(matplotlib not installed; FDP+ curve skipped.)")

    return res