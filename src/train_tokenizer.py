#!/usr/bin/env python3
"""
src/train_tokenizer.py  (step 5, part 1)

Trains a byte-level BPE tokenizer (Hugging Face `tokenizers`) on the TRAINING
texts only. Validation works are never seen, so there is no leakage.

Design choices
- Byte-level BPE (GPT-2 style): every byte is in the base alphabet, so any
  UTF-8 text encodes with no <unk> and decode(encode(x)) == x exactly.
- No normalizer: the clean corpus is already NFC UTF-8; normalizing inside the
  tokenizer would break the exact round-trip.
- One special token, <|endoftext|>, used by encode.py as a document separator.
- vocab_size defaults to 8192 (< 65536, so token ids fit in uint16).
- Inputs are sorted and the trainer is single-config, so reruns give the same
  tokenizer; its SHA-256 is recorded in meta.json by encode.py.

Usage
    python src/train_tokenizer.py                           # full train split
    python src/train_tokenizer.py --inputs data/clean/train/donquijote.txt
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = BASE_DIR / "data" / "clean" / "train"
DEFAULT_OUT = BASE_DIR / "data" / "tokenized" / "tokenizer.json"
EOT = "<|endoftext|>"


def collect_files(inputs):
    """Expand files/directories into a sorted, de-duplicated list of .txt files."""
    files = []
    for p in map(Path, inputs):
        if p.is_dir():
            files.extend(sorted(p.glob("*.txt")))
        elif p.is_file():
            files.append(p)
        else:
            sys.exit(f"[ERROR] input not found: {p}")
    files = sorted(set(f.resolve() for f in files))
    if not files:
        sys.exit("[ERROR] no .txt files found in inputs")
    for f in files:
        if "/val/" in f.as_posix():
            sys.exit(f"[ERROR] refusing to train the tokenizer on validation text: {f}")
    return files


def build_tokenizer():
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    return tok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", default=[str(DEFAULT_INPUT)],
                    help="training .txt files and/or directories (default: data/clean/train)")
    ap.add_argument("--vocab_size", type=int, default=8192)
    ap.add_argument("--min_frequency", type=int, default=2)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    if args.vocab_size >= 2**16:
        sys.exit("[ERROR] vocab_size must be < 65536 so ids fit in uint16")

    files = collect_files(args.inputs)
    total_mb = sum(f.stat().st_size for f in files) / 2**20
    print(f"[INFO] training BPE (vocab={args.vocab_size}) on {len(files)} file(s), {total_mb:.2f} MB")

    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    t0 = time.time()
    tok.train([str(f) for f in files], trainer)
    print(f"[INFO] trained in {time.time() - t0:.1f}s, final vocab size {tok.get_vocab_size()}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out))
    sha = hashlib.sha256(out.read_bytes()).hexdigest()

    # Small provenance file next to the tokenizer (encode.py copies this into meta.json).
    prov = {
        "vocab_size": tok.get_vocab_size(),
        "requested_vocab_size": args.vocab_size,
        "min_frequency": args.min_frequency,
        "eot_token": EOT,
        "eot_id": tok.token_to_id(EOT),
        "trained_on": [f.name for f in files],
        "trained_on_mb": round(total_mb, 3),
        "tokenizer_sha256": sha,
    }
    out.with_name("tokenizer_provenance.json").write_text(json.dumps(prov, indent=2, ensure_ascii=False))

    # Quick sanity check on a Spanish sentence.
    s = "En un lugar de la Mancha, de cuyo nombre no quiero acordarme…"
    ids = tok.encode(s).ids
    assert tok.decode(ids) == s, "round-trip failed on sample sentence"
    print(f"[INFO] sample: {len(s)} chars -> {len(ids)} tokens")
    print(f"[DONE] wrote {out} (sha256 {sha[:12]}…)")


if __name__ == "__main__":
    main()
