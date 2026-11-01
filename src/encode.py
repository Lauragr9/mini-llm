#!/usr/bin/env python3
"""
src/encode.py  (step 5, part 2)

Encodes the clean corpus with the trained tokenizer into flat uint16 token
streams that train.py reads with np.memmap.

Outputs (default: data/tokenized/)
    train.bin   all training works, each followed by <|endoftext|>
    val.bin     all held-out works, same format
    tiny.bin    first --tiny_tokens tokens of train.bin (smoke tests)
    meta.json   vocab size, dtype, EOT id, token counts, provenance
    (tokenizer.json is written there by train_tokenizer.py)

Format: raw little-endian uint16, no header. Load with
    np.memmap("data/tokenized/train.bin", dtype=np.uint16, mode="r")

Every document is checked on the way: decode(encode(doc)) must equal doc,
otherwise the script exits with an error and writes nothing.

Usage
    python src/encode.py
    python src/encode.py --train data/clean/train/donquijote.txt --val data/clean/val
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tokenizers
from tokenizers import Tokenizer

BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN = BASE_DIR / "data" / "clean"
OUT = BASE_DIR / "data" / "tokenized"
EOT = "<|endoftext|>"
DTYPE = np.uint16


def collect_files(inputs):
    files = []
    for p in map(Path, inputs):
        if p.is_dir():
            files.extend(sorted(p.glob("*.txt")))
        elif p.is_file():
            files.append(p)
        else:
            sys.exit(f"[ERROR] input not found: {p}")
    files = sorted(set(f.resolve() for f in files), key=lambda f: f.name)
    if not files:
        sys.exit(f"[ERROR] no .txt files in {inputs}")
    return files


def encode_split(tok, files, eot_id, name):
    """Encode each document, verify round-trip, return (np.ndarray, per-doc stats)."""
    chunks, docs = [], []
    for f in files:
        text = f.read_text(encoding="utf-8")
        ids = tok.encode(text).ids
        if tok.decode(ids) != text:
            sys.exit(f"[ERROR] round-trip mismatch in {name}/{f.name}")
        if eot_id in ids:
            sys.exit(f"[ERROR] {f.name} contains the literal string {EOT}")
        chunks.append(np.asarray(ids, dtype=DTYPE))
        chunks.append(np.asarray([eot_id], dtype=DTYPE))
        docs.append({
            "file": f.name,
            "chars": len(text),
            "bytes_utf8": len(text.encode("utf-8")),
            "tokens": len(ids),
        })
        print(f"  [{name}] {f.name:40s} {len(text):>10,} chars -> {len(ids):>9,} tokens")
    arr = np.concatenate(chunks)
    return arr, docs


def write_bin(arr, path):
    arr.astype(DTYPE).tofile(path)
    check = np.memmap(path, dtype=DTYPE, mode="r")
    assert check.shape == arr.shape and np.array_equal(check[:1000], arr[:1000])
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_summary(arr, docs, path, sha):
    chars = sum(d["chars"] for d in docs)
    return {
        "file": path.name,
        "tokens": int(arr.size),                     # includes one EOT per document
        "documents": len(docs),
        "chars": chars,
        "bytes_utf8": sum(d["bytes_utf8"] for d in docs),
        "chars_per_token": round(chars / max(1, arr.size - len(docs)), 3),
        "sha256": sha,
        "docs": docs,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", default=[str(CLEAN / "train")])
    ap.add_argument("--val", nargs="+", default=[str(CLEAN / "val")])
    ap.add_argument("--out_dir", default=str(OUT))
    ap.add_argument("--tokenizer", default=None, help="default: <out_dir>/tokenizer.json")
    ap.add_argument("--tiny_tokens", type=int, default=1_500_000,
                    help="tokens in tiny.bin (default 1.5M = 3 MB)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok_path = Path(args.tokenizer) if args.tokenizer else out / "tokenizer.json"
    if not tok_path.exists():
        sys.exit(f"[ERROR] {tok_path} not found - run src/train_tokenizer.py first")
    tok = Tokenizer.from_file(str(tok_path))
    vocab = tok.get_vocab_size()
    eot_id = tok.token_to_id(EOT)
    if eot_id is None:
        sys.exit(f"[ERROR] tokenizer has no {EOT} token")
    if vocab > np.iinfo(DTYPE).max + 1:
        sys.exit(f"[ERROR] vocab {vocab} does not fit in uint16")

    train_files, val_files = collect_files(args.train), collect_files(args.val)
    overlap = {f.name for f in train_files} & {f.name for f in val_files}
    if overlap:
        sys.exit(f"[ERROR] works in both train and val: {sorted(overlap)}")

    t0 = time.time()
    print(f"[INFO] tokenizer {tok_path.name}: vocab {vocab}, EOT id {eot_id}")
    train, train_docs = encode_split(tok, train_files, eot_id, "train")
    val, val_docs = encode_split(tok, val_files, eot_id, "val")
    tiny = train[: args.tiny_tokens]

    # Write everything only after all round-trip checks have passed.
    shas = {n: write_bin(a, out / f"{n}.bin") for n, a in [("train", train), ("val", val), ("tiny", tiny)]}

    prov_path = tok_path.with_name("tokenizer_provenance.json")
    meta = {
        "vocab_size": vocab,
        "dtype": "uint16",
        "byte_order": "little",
        "eot_token": EOT,
        "eot_id": eot_id,
        "tokenizer": tok_path.name,
        "tokenizer_sha256": hashlib.sha256(tok_path.read_bytes()).hexdigest(),
        "tokenizer_training": json.loads(prov_path.read_text()) if prov_path.exists() else None,
        "splits": {
            "train": split_summary(train, train_docs, out / "train.bin", shas["train"]),
            "val": split_summary(val, val_docs, out / "val.bin", shas["val"]),
            "tiny": {"file": "tiny.bin", "tokens": int(tiny.size), "source": "first tokens of train.bin",
                     "sha256": shas["tiny"]},
        },
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "tokenizers": tokenizers.__version__},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print("-" * 60)
    for n in ("train", "val"):
        s = meta["splits"][n]
        print(f"{n:5s}: {s['tokens']:>11,} tokens  {s['documents']:>3} docs  "
              f"{s['bytes_utf8'] / 2**20:6.2f} MB text  {s['chars_per_token']} chars/token")
    print(f"tiny : {tiny.size:>11,} tokens  ({tiny.nbytes / 2**20:.2f} MB)")
    print(f"[DONE] wrote train.bin, val.bin, tiny.bin, meta.json to {out} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
