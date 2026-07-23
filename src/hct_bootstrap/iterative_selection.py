import random
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from sklearn.utils import resample

from .core import higher_criticism
from .null import split_half_null, resample_null


def _hc_select_words(text_a, name_a, text_b, name_b) -> set:
    if not text_a.strip() or not text_b.strip():
        return set()
    _, hc_df, _ = higher_criticism(text_a, name_a, text_b, name_b)
    if hc_df is None or hc_df.empty or 'word' not in hc_df.columns:
        return set()
    return set(hc_df['word'])


def _bootstrap_one_iteration(docs_a, name_a, docs_b, name_b,
                             coupled, seed_a, seed_b) -> set:
    if coupled:
        if len(docs_a) != len(docs_b):
            raise ValueError("Coupled bootstrap requires equal-length lists.")
        n_pairs = len(docs_a)
        idx = resample(list(range(n_pairs)), n_samples=n_pairs, replace=True,
                       random_state=seed_a)
        sample_a = [docs_a[i] for i in idx]
        sample_b = [docs_b[i] for i in idx]
    else:
        sample_a = resample(docs_a, n_samples=len(docs_a), replace=True,
                            random_state=seed_a)
        sample_b = resample(docs_b, n_samples=len(docs_b), replace=True,
                            random_state=seed_b)
    text_a = '\n\n'.join(d for d in sample_a if d)
    text_b = '\n\n'.join(d for d in sample_b if d)
    return _hc_select_words(text_a, name_a, text_b, name_b)


def _build_null_pairs(docs_a, docs_b, null_method, seed,
                      user_null_pairs=None):
    """
    Return a list of null comparison pairs, each a 4-tuple
        (docs_1, docs_2, label, coupled_flag).

    For 'split'/'resample', the null pair is built from a SINGLE input corpus
    and is always uncoupled (coupled_flag=False). For 'user', the caller passes
    `user_null_pairs` already built (each original corpus vs its user-supplied
    null corpus), carrying its own coupled_flag per pair.
    """
    if null_method == 'user':
        if not user_null_pairs:
            raise ValueError("null_method='user' requires user_null_pairs.")
        return user_null_pairs

    rng = random.Random(seed)
    if null_method == 'split':
        a1, a2 = split_half_null(docs_a, seed=rng.randint(0, 2**32 - 1))
        b1, b2 = split_half_null(docs_b, seed=rng.randint(0, 2**32 - 1))
    elif null_method == 'resample':
        a1, a2 = resample_null(docs_a, rng.randint(0, 2**32 - 1),
                               rng.randint(0, 2**32 - 1))
        b1, b2 = resample_null(docs_b, rng.randint(0, 2**32 - 1),
                               rng.randint(0, 2**32 - 1))
    else:
        raise ValueError(f"Unknown null_method '{null_method}'. "
                         f"Use 'split', 'resample', or 'user'.")
    return [(a1, a2, 'null_A', False), (b1, b2, 'null_B', False)]


class _CumulativeIntersection:
    """
    Maintains a running intersection of bootstrap-selected word sets for one
    corpus pair. Calling .add() draws ONE fresh bootstrap resample, runs HCT,
    and intersects the result into the running set. After k calls, .survivors
    holds the words selected in ALL k draws so far.

    This is the cumulative design: N(n) reuses draws 1..n-1 and adds one new
    draw at step n, so N(n) is monotonically non-increasing in n.
    """
    def __init__(self, docs_a, name_a, docs_b, name_b, coupled, rng):
        self.docs_a = docs_a
        self.name_a = name_a
        self.docs_b = docs_b
        self.name_b = name_b
        self.coupled = coupled
        self.rng = rng
        self.survivors = None   # None until first draw; then a set

    def add(self) -> set:
        # Once the intersection is empty it stays empty; skip further HC work.
        if self.survivors is not None and len(self.survivors) == 0:
            return self.survivors
        sel = _bootstrap_one_iteration(
            self.docs_a, self.name_a, self.docs_b, self.name_b, self.coupled,
            self.rng.randint(0, 2**32 - 1), self.rng.randint(0, 2**32 - 1))
        if self.survivors is None:
            self.survivors = set(sel)
        else:
            self.survivors &= sel
        return self.survivors


# --------------------------------------------------------------------------- #
# Main procedure: cumulative intersection over n draws
# --------------------------------------------------------------------------- #

def discriminate_until_rule(
    test_docs_a, name_a,
    test_docs_b, name_b,
    coupled,
    hct_words,
    q: Optional[float] = 0.05,
    null_method: str = 'split',
    user_null_pairs=None,
    max_t: int = 1000,
    random_seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Adaptive BS-HCT using a CUMULATIVE intersection over n bootstrap draws.

    We draw a growing sequence of bootstrap resamples. At step n:
      * the test survivors are the words selected by HCT in ALL of the first n
        draws (intersection of draws 1..n) -> count N(n);
      * the same cumulative intersection is maintained for each within-corpus
        null (always uncoupled), and N0(n) is the pointwise maximum across
        nulls;
      * FDP+(n) = (1 + N0(n)) / max(N(n), 1).

    Because step n reuses draws 1..n-1 and adds one new draw, both N(n) and
    N0(n) are monotonically NON-INCREASING in n: intersecting one more set can
    only remove words, never add them. This is cheaper than redrawing all n
    resamples each step (cost is linear in n, not quadratic) and gives a clean
    monotone survival curve.

    Stopping rule:
      * q is None    -> STRICT: stop at the first n with N0(n) == 0.
      * q is a float -> FDR:    stop at the first n with FDP+(n) <= q.
    The chosen n is the number of bootstrap iterations; max_t caps the search.

    null_method:
      * 'split' / 'resample' -> nulls built from the input corpora (uncoupled).
      * 'user' -> caller supplies `user_null_pairs`.

    Final selection = {test words surviving all n_stop draws} ∩ HCT.
    """
    if q is not None and q < 0:
        raise ValueError("q must be None (strict rule) or a non-negative float.")

    rule = 'strict' if q is None else 'fdr'
    rng = random.Random(random_seed)

    null_pairs = _build_null_pairs(test_docs_a, test_docs_b, null_method,
                                   seed=rng.randint(0, 2**32 - 1),
                                   user_null_pairs=user_null_pairs)

    # Set up the cumulative-intersection accumulators (test + one per null).
    test_acc = _CumulativeIntersection(
        test_docs_a, name_a, test_docs_b, name_b, coupled, rng)
    null_accs = [
        _CumulativeIntersection(n1, 'null_1', n2, 'null_2', n_coupled, rng)
        for (n1, n2, _lbl, n_coupled) in null_pairs
    ]

    t_stop = None
    fdp_at_stop = None
    last_test_survivors: set = set()
    last_null_survivors: List[set] = [set() for _ in null_pairs]
    fdp_curve: List[Tuple[int, int, int, float]] = []

    t = 0
    for t in range(1, max_t + 1):
        # Add ONE new draw to each cumulative intersection.
        test_surv = test_acc.add()
        N_test = len(test_surv)

        null_surv = [acc.add() for acc in null_accs]
        N_null_max = max((len(s) for s in null_surv), default=0)

        fdp_plus = (1 + N_null_max) / max(N_test, 1)
        fdp_curve.append((t, N_test, N_null_max, fdp_plus))
        last_test_survivors = set(test_surv)
        last_null_survivors = [set(s) for s in null_surv]

        if verbose:
            print(f"  t={t:5d} | N_test={N_test:5d} | N0_max={N_null_max:4d} "
                  f"| FDP+={fdp_plus:.4f}", flush=True)

        if rule == 'strict':
            if N_null_max == 0:
                t_stop, fdp_at_stop = t, fdp_plus
                break
        else:
            if fdp_plus <= q:
                t_stop, fdp_at_stop = t, fdp_plus
                break

    empty_reason = None
    if t_stop is None:
        selected: List[str] = []
        if rule == 'strict':
            empty_reason = (f"No null word set ever emptied (N0(t) > 0 for all "
                            f"t) within max_t={max_t}.")
        else:
            empty_reason = (f"FDP+(t) never fell to q={q} within max_t={max_t}; "
                            f"the achievable floor is ~1/N_test.")
    else:
        selected = sorted(set(last_test_survivors) & set(hct_words))
        if not selected:
            empty_reason = (f"Rule fired at t={t_stop} but no surviving test "
                            f"words were in the HCT set.")

    def _surv_df(words):
        return (pd.DataFrame(columns=['word']) if not words
                else pd.DataFrame({'word': sorted(words)}))

    return {
        "selected_words": selected,
        "t_stop": t_stop,
        "n_rule": t_stop,
        "fdp_at_stop": fdp_at_stop,
        "q": q,
        "rule": rule,
        "null_method": null_method,
        "iterations_run": t,
        "empty_reason": empty_reason,
        "fdp_curve": fdp_curve,
        "test_survivors_df": _surv_df(last_test_survivors),
        "null_survivors_dfs": [_surv_df(s) for s in last_null_survivors],
    }


# --------------------------------------------------------------------------- #
# Plotting: FDP+(t) and survival counts N(t), N0(t) over t
# --------------------------------------------------------------------------- #

def plot_fdp_curve(result: Dict[str, Any], ax=None):
    """
    Plot N(n), N0(n), and FDP+(n) over n from the cumulative-intersection run,
    marking the stopping point. Uses result['fdp_curve'].
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        print("matplotlib not installed; cannot plot.")
        return None

    curve = result.get("fdp_curve", [])
    if not curve:
        print("No fdp_curve data to plot.")
        return None

    ts = [c[0] for c in curve]
    N_test = [c[1] for c in curve]
    N0 = [c[2] for c in curve]
    fdp = [c[3] for c in curve]
    t_stop = result.get("t_stop")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(ts, N_test, color='#6a0dad', linewidth=2,
            label=r'$\hat{N}(n)$ (test survivors)')
    ax.plot(ts, N0, color='red', linewidth=2, linestyle='--',
            label=r'$\hat{N}_0(n)$ (max null survivors)')
    ax.set_xlabel("n (number of bootstrap iterations)", fontsize=13)
    ax.set_ylabel("Number of surviving words", fontsize=13)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax.twinx()
    ax2.plot(ts, fdp, color='green', linewidth=1.6, alpha=0.7,
             label=r'$\mathrm{FDP}^+(n)$')
    ax2.set_ylabel(r'$\mathrm{FDP}^+(n)$', fontsize=13, color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    if t_stop is not None:
        ax.axvline(x=t_stop, color='gray', linestyle=':', linewidth=1.8)
        ax.annotate(rf'$n_{{stop}}={t_stop}$', xy=(t_stop, max(N_test) * 0.8),
                    xytext=(t_stop, max(N_test) * 0.9), fontsize=12, ha='center')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11)
    ax.set_title("BS-HCT: survival counts and FDP+ over n", fontsize=15, pad=12)
    return ax