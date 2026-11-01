#!/usr/bin/env python3
"""
src/test_roundtrip.py  (step 5, verification)

End-to-end check of the files train.py will consume. Reads the .bin files back
from disk with np.memmap (exactly as train.py does) and verifies:

  1. sizes match meta.json and every id < vocab_size
  2. tiny.bin is a prefix of train.bin
  3. splitting train.bin / val.bin on <|endoftext|> and decoding each piece
     gives back the clean source file byte-for-byte
  4. tricky strings (accents, ñ, ¿¡, «», em-dash, emoji, whitespace runs) round-trip

Exit code 0 = all passed. Works as a plain script or under pytest.

Usage
    python src/test_roundtrip.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

BASE_DIR = Path(__file__).resolve().parent.parent
TOK_DIR = BASE_DIR / "data" / "tokenized"
CLEAN = BASE_DIR / "data" / "clean"

EDGE_CASES = [
    "",
    "En un lugar de la Mancha, de cuyo nombre no quiero acordarme",
    "¿Qué?  ¡Ay!  —dijo Sancho— «vuestra merced» … ñandú, pingüino, ÁÉÍÓÚÜÑ",
    "línea 1\n\n\n  línea 2\t\ttab\r\nfin   ",
    "naïve café 東京 🙂 € ½",
]


def load(split):
    return np.memmap(TOK_DIR / f"{split}.bin", dtype=np.uint16, mode="r")


def test_roundtrip():
    meta = json.loads((TOK_DIR / "meta.json").read_text())
    tok = Tokenizer.from_file(str(TOK_DIR / meta["tokenizer"]))
    eot, vocab = meta["eot_id"], meta["vocab_size"]
    assert meta["dtype"] == "uint16" and vocab == tok.get_vocab_size()

    data = {s: load(s) for s in ("train", "val", "tiny")}
    for s, arr in data.items():
        assert arr.size == meta["splits"][s]["tokens"], f"{s}.bin size != meta.json"
        assert int(arr.max()) < vocab, f"{s}.bin has id >= vocab_size"
        print(f"[OK] {s}.bin: {arr.size:,} tokens, max id {int(arr.max())} < {vocab}")

    assert np.array_equal(data["tiny"], data["train"][: data["tiny"].size]), "tiny.bin is not a prefix of train.bin"
    print("[OK] tiny.bin is a prefix of train.bin")

    for split in ("train", "val"):
        arr = np.asarray(data[split])
        cuts = np.flatnonzero(arr == eot)
        docs = meta["splits"][split]["docs"]
        assert len(cuts) == len(docs), f"{split}: {len(cuts)} EOT markers but {len(docs)} docs"
        start = 0
        for cut, d in zip(cuts, docs):
            original = (CLEAN / split / d["file"]).read_text(encoding="utf-8")
            decoded = tok.decode(arr[start:cut].tolist())
            assert decoded == original, f"round-trip mismatch: {split}/{d['file']}"
            start = cut + 1
        print(f"[OK] {split}: all {len(docs)} documents decode back to the clean text exactly")

    for s in EDGE_CASES:
        assert tok.decode(tok.encode(s).ids) == s, f"edge case failed: {s!r}"
    print(f"[OK] {len(EDGE_CASES)} edge-case strings round-trip")


if __name__ == "__main__":
    try:
        test_roundtrip()
    except AssertionError as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        sys.exit(1)
    print("[PASS] all round-trip checks passed")
