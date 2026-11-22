"""Token-budget micro-batching for training, and length-sorted batching for evaluation.

Why this exists. The trainer used a fixed micro_batch_size of 8 with padding to the longest
molecule in the micro-batch. Measured on ChemFM-1B / NR-ER: token lengths are mean 35, p99 165,
max 342, so 57 % of the tokens in a micro-batch were padding and the GPU ran at 37.5 molecules/s.
Grouping the SAME effective batch by a token budget instead reaches 90.9 molecules/s, a 2.4x
speed-up, at 16-30 GB depending on the budget.

The split is EXACT, not an approximation. One optimiser step covers one effective batch; the loss
is the sum of that batch's per-sample losses divided by the batch size, and a sum does not depend
on how the batch is partitioned for the backward pass. Batch membership, the optimiser step and
the normaliser are all unchanged; only the grouping inside the step changes.
"""
from __future__ import annotations

import numpy as np


def token_lengths(backbone, smiles) -> np.ndarray:
    """Prompt token length per molecule, canonical form (augmentation preserves atom count)."""
    tok, fmt = backbone.tokenizer, backbone.prompt_format
    add_special = not backbone.kind.startswith("chemfm")
    return np.array([len(tok(fmt.format(smiles=s), add_special_tokens=add_special)["input_ids"])
                     for s in smiles], dtype=np.int32)


def budget_micro_batches(indices, lengths, token_budget: int):
    """Partition one effective batch into padded-token-bounded groups, longest first.

    Sorting descending means the first member of a group is its longest, so
    `max_len * group_size` is exact as the group grows and the padded footprint of every group
    stays under `token_budget`. A molecule longer than the budget forms its own group rather than
    being dropped.
    """
    idx = sorted(indices, key=lambda i: (-int(lengths[i]), int(i)))
    groups, cur, cur_max = [], [], 0
    for i in idx:
        li = int(lengths[i])
        m = max(cur_max, li)
        if cur and m * (len(cur) + 1) > token_budget:
            groups.append(cur)
            cur, cur_max = [i], li
        else:
            cur.append(i)
            cur_max = m
    if cur:
        groups.append(cur)
    return groups


def eval_groups(backbone, smiles, token_budget: int):
    """Length-sorted evaluation groups under the same budget. Returns lists of positions."""
    lens = token_lengths(backbone, smiles)
    order = np.argsort(-lens, kind="stable")
    groups, cur, cur_max = [], [], 0
    for i in order:
        li = int(lens[i])
        m = max(cur_max, li)
        if cur and m * (len(cur) + 1) > token_budget:
            groups.append(cur)
            cur, cur_max = [int(i)], li
        else:
            cur.append(int(i))
            cur_max = m
    if cur:
        groups.append(cur)
    return groups, lens


BUCKET_POOL = 1024        # measured choice; see the docstring below


def bucketed_batches(n, lengths, batch_size, rng, pool=BUCKET_POOL):
    """Form an epoch's batches by shuffling, then sorting by length INSIDE each pool.

    Why: with a plain random permutation a batch of 32 is padded to its longest member, and on
    NR-ER (token lengths mean 35, p99 165, max 342) that wasted 67 % of the tokens in every
    forward. Sorting inside a pool makes each batch length-homogeneous.

    Measured on ChemFM-1B, token budget 4096, median of 3 interleaved repeats:

        random permutation   95.4 mol/s   padding 0.669   1.00x
        pool  256           144.0 mol/s   padding 0.256   1.51x
        pool  512           157.1 mol/s   padding 0.159   1.65x
        pool 1024           180.1 mol/s   padding 0.133   1.89x     <- chosen
        full-epoch sort     176.6 mol/s   padding 0.096   1.85x

    Pool 1024 is chosen over a full sort because they are the same speed while the full sort
    produces the SAME batches every epoch — it removes epoch-to-epoch randomness entirely. With
    1024 over ~1,100-1,500 molecules per client there are only a handful of pools, which pool a
    molecule lands in changes every round, and batches stay length-homogeneous only within a pool.

    This is NOT free: batch composition now correlates with molecule size, so a gradient step is
    no longer an i.i.d. sample of the client's data and training results change. Every arm must
    use the same setting for arm-to-arm comparisons to remain valid, and numbers produced before
    and after this change must not be mixed.
    """
    o = rng.permutation(n)
    out = []
    for s in range(0, n, pool):
        blk = o[s:s + pool]
        blk = blk[np.argsort(np.asarray(lengths)[blk], kind="stable")]
        out += [list(blk[i:i + batch_size]) for i in range(0, len(blk), batch_size)]
    return [out[i] for i in rng.permutation(len(out))]
