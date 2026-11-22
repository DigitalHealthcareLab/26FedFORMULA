"""Shared guards for the experiment scripts.

Each one exists because the corresponding mistake actually cost time in this project:

  safe_merge        a merge whose frames shared a column silently produced `is_flip_x`, and the
                    next attribute access either crashed or read the wrong column.
  require_free_vram two chains fired on the same completion signal, claimed one card together and
                    OOMed; one of them then hung for 30 minutes waiting on a dead worker.
  tagged_path       a learning-curve run wrote `T0__best_state.pt` over the full-data checkpoint of
                    the same arm, and the wrong model was evaluated and reported.
  col               `r.T` is DataFrame.transpose and `r.where` is DataFrame.where, so attribute
                    access on a row silently returns a bound method for those column names.
"""
from __future__ import annotations

import os
import subprocess


def safe_merge(left, right, on, want=None, how="left"):
    """Merge, taking from `right` only the columns `left` does not already have.

    Raises if a wanted column is missing afterwards. When there is no overlap this is exactly a
    plain merge, so it never changes an existing result; when there is one it fails loudly instead
    of inventing `_x` / `_y` names.
    """
    keys = [on] if isinstance(on, str) else list(on)
    take = keys + [c for c in right.columns if c not in keys and c not in left.columns]
    dropped = [c for c in right.columns if c not in take]
    out = left.merge(right[take], on=on, how=how)
    if want:
        missing = [c for c in want if c not in out.columns]
        if missing:
            raise KeyError(f"safe_merge: {missing} absent after merge "
                           f"(dropped as already present: {dropped})")
    return out


def col(row, name):
    """Read a column from a row by label. Use instead of `row.name` attribute access whenever the
    column could shadow a pandas attribute (T, where, index, size, name, values, ...)."""
    return row[name]


def free_vram_mb(device_index: int) -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                          "--format=csv,noheader,nounits", "-i", str(device_index)],
                         capture_output=True, text=True, check=True)
    return int(out.stdout.strip().splitlines()[0])


def require_free_vram(devices, need_mb: int, what: str = "this job"):
    """Refuse to start when a target card cannot hold the job. Returns the measured free MB.

    `devices` is 'cuda:0' or a list/comma string of them. Set TOX21_SKIP_VRAM_CHECK=1 to bypass,
    which should only be done when the caller has already reserved the card.
    """
    if os.environ.get("TOX21_SKIP_VRAM_CHECK") == "1":
        return {}
    if isinstance(devices, str):
        devices = devices.split(",")
    free = {}
    for d in devices:
        d = d.strip()
        if not d.startswith("cuda"):
            continue
        i = int(d.split(":")[1]) if ":" in d else 0
        free[d] = free_vram_mb(i)
    short = {d: m for d, m in free.items() if m < need_mb}
    if short:
        raise SystemExit(f"refusing to start {what}: {short} free MB, needs {need_mb} MB per card. "
                         f"Another job is holding the GPU; wait for it or pick another device.")
    return free


def tagged_path(out_dir: str, tag: str, suffix: str) -> str:
    """Every artefact path carries the run tag, so two runs of one arm cannot overwrite each other."""
    assert tag and "__" not in suffix.split(".")[0][:2], "suffix should not start a new tag"
    return os.path.join(out_dir, f"{tag}__{suffix}")


# ---------------------------------------------------------------- atomic output
# A process killed mid-write leaves a truncated CSV or checkpoint behind. Nothing downstream can
# tell that from a complete one, so a later stage reads a partial run as if it finished. That is
# what happened to round_metrics__T5__R10.csv (6 of 7 rounds) and __T8__ (5 of 7): both were
# picked up by the report generator as complete runs. Write through a temp file and rename, so a
# path either does not exist or is whole.

import json as _json
import tempfile as _tempfile


def _atomic(path, write_fn):
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = _tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=os.path.basename(path))
    os.close(fd)
    try:
        write_fn(tmp)
        os.replace(tmp, path)            # atomic within a filesystem
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def atomic_to_csv(df, path, **kw):
    kw.setdefault("index", False)
    return _atomic(path, lambda t: df.to_csv(t, **kw))


def atomic_to_parquet(df, path, **kw):
    kw.setdefault("index", False)
    return _atomic(path, lambda t: df.to_parquet(t, **kw))


def atomic_json(obj, path, **kw):
    kw.setdefault("indent", 2)
    return _atomic(path, lambda t: _json.dump(obj, open(t, "w"), **kw))


def atomic_text(text, path):
    return _atomic(path, lambda t: open(t, "w").write(text))


def atomic_torch_save(obj, path):
    import torch
    return _atomic(path, lambda t: torch.save(obj, t))


# ---------------------------------------------------------------- completion markers
# A run's own metadata file is written last, so its presence already means "finished". Per-round
# artefacts have no such guarantee, hence an explicit marker that consumers can require.

def mark_done(path, payload=None):
    """Write `<path>.done` after the artefact itself is complete."""
    atomic_json(payload or {"complete": True}, f"{path}.done")
    return f"{path}.done"


def is_done(path) -> bool:
    return os.path.exists(f"{path}.done")


def require_done(path, what=""):
    if not is_done(path):
        raise SystemExit(f"refusing to use an unfinished artefact: {path} has no .done marker "
                         f"({what}). Re-run the stage that produces it.")
    return path


def read_complete_csv(path, expected_rows=None, what=""):
    """Read a CSV only if it is known-complete: a .done marker, or an exact row count."""
    import pandas as pd
    d = pd.read_csv(path)
    if expected_rows is not None and len(d) != expected_rows:
        raise SystemExit(f"{path} has {len(d)} rows, expected {expected_rows} — this looks like a "
                         f"run that was interrupted ({what}). Move it aside or re-run the stage.")
    return d


# ---------------------------------------------------------------- environment routing
# Three virtualenvs exist and they are NOT interchangeable:
#   benchmark_v1/envs/chemfm            torch + transformers + peft + rdkit  -> training, eval
#   cleansing_icl_pilot_v1/envs/icl     torch + transformers, NO peft        -> LLM scoring only
#   postpilot_v5/envs/explain           exmol + rdkit, NO torch              -> STONED generation
# Running a training script under `icl` dies on `import peft`; running anything torch-based under
# `explain` dies on `import torch`. Both fail minutes in, after the scheduler has already moved on.

ENV_REQUIREMENTS = {
    "train": ["torch", "transformers", "peft", "rdkit", "pandas", "sklearn"],
    # MLP/D-MPNN need no transformers and no peft, and D-MPNN's chemprop dependency only
    # resolves in the `chemprop` venv, which has neither. Demanding the full "train" set there
    # would reject the correct interpreter.
    "train_small": ["torch", "rdkit", "pandas", "sklearn"],
    "score": ["torch", "transformers", "pandas"],
    "explain": ["rdkit", "exmol", "pandas"],
    "analysis": ["pandas", "numpy", "sklearn"],
    "figures": ["pandas", "matplotlib"],
}


def require_env(kind: str, what: str = ""):
    """Fail immediately, with the interpreter path, if this venv cannot run this kind of job."""
    import importlib
    import sys as _sys
    need = ENV_REQUIREMENTS.get(kind)
    if need is None:
        raise ValueError(f"unknown env kind {kind!r}; expected one of {sorted(ENV_REQUIREMENTS)}")
    missing = []
    for m in need:
        try:
            importlib.import_module(m)
        except Exception as e:
            missing.append(f"{m} ({type(e).__name__})")
    if missing:
        raise SystemExit(
            f"wrong interpreter for a '{kind}' job{(' — ' + what) if what else ''}.\n"
            f"  running: {_sys.executable}\n"
            f"  missing: {', '.join(missing)}\n"
            f"  train/eval -> benchmark_v1/envs/chemfm/bin/python\n"
            f"  MLP/D-MPNN -> benchmark_v1/envs/chemprop/bin/python\n"
            f"  LLM scoring -> experiments/cleansing_icl_pilot_v1/envs/icl/bin/python\n"
            f"  STONED/exmol -> experiments/postpilot_v5/envs/explain/bin/python")
    return True
