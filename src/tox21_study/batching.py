"""Batching helpers for the no-gradient passes.

SMILES lengths in this cohort are very uneven: median 26 tokens, 95th percentile 81, longest 274.
Batching in arbitrary order pads every sequence up to the longest one in its batch, so most of the
compute is spent on padding. Sorting by length before batching removes that waste. Measured on
ChemFM-3B, one forward over 1,024 training molecules:

    arbitrary order, batch 16        144 molecules/s
    arbitrary order, batch 32        110
    length-sorted,   batch 32        287

Only the gradient-free passes use this. Training order stays shuffled, because sorting there would
correlate batch composition with molecule size and change the optimisation itself.
"""
from __future__ import annotations

import numpy as np


def length_sorted_batches(lengths, batch_size: int):
    """Yields (positions,) batches of similar length. Positions index the original sequence, so the
    caller writes results straight back into their original slots and order is preserved."""
    order = np.argsort(np.asarray(lengths), kind="stable")
    for s in range(0, len(order), batch_size):
        yield order[s:s + batch_size]


def token_lengths(backbone, smiles) -> np.ndarray:
    """Prompt-formatted token count per molecule, using the backbone's own tokenizer."""
    tok = backbone.tokenizer
    special = not backbone.kind.startswith("chemfm")
    return np.array([len(tok(backbone.prompt_format.format(smiles=s),
                             add_special_tokens=special)["input_ids"]) for s in smiles])


def eval_batches(backbone, smiles, batch_size: int, sort_by_length: bool = True):
    """Batches for a no-gradient pass over `smiles`, longest-together when sorting is on."""
    n = len(smiles)
    if not sort_by_length or n <= batch_size:
        for s in range(0, n, batch_size):
            yield np.arange(s, min(s + batch_size, n))
        return
    yield from length_sorted_batches(token_lengths(backbone, smiles), batch_size)
