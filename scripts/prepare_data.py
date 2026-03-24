"""
Download and tokenize a dataset into memmap format for training.

Usage:
    python scripts/prepare_data.py --dataset openwebtext --out-dir data/openwebtext
    python scripts/prepare_data.py --dataset wikitext --out-dir data/wikitext
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ash.data.tokenizer import Tokenizer


def prepare_openwebtext(out_dir: Path):
    from datasets import load_dataset

    print("Loading OpenWebText...")
    dataset = load_dataset("openwebtext", split="train", trust_remote_code=True)
    tokenizer = Tokenizer()

    # Split 90/10
    split = dataset.train_test_split(test_size=0.1, seed=42)

    for name, ds in [("train", split["train"]), ("val", split["test"])]:
        print(f"Tokenizing {name} ({len(ds)} documents)...")
        all_tokens = []
        for doc in tqdm(ds):
            tokens = tokenizer.encode(doc["text"])
            tokens.append(tokenizer.eot_token)
            all_tokens.extend(tokens)

        arr = np.array(all_tokens, dtype=np.uint16)
        out_path = out_dir / f"{name}.bin"
        arr.tofile(str(out_path))
        print(f"  {name}: {len(arr):,} tokens -> {out_path}")


def prepare_wikitext(out_dir: Path):
    from datasets import load_dataset

    print("Loading WikiText-103...")
    tokenizer = Tokenizer()

    for name, split_name in [("train", "train"), ("val", "validation")]:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split_name)
        print(f"Tokenizing {name} ({len(ds)} lines)...")
        all_tokens = []
        for row in tqdm(ds):
            text = row["text"].strip()
            if text:
                tokens = tokenizer.encode(text)
                tokens.append(tokenizer.eot_token)
                all_tokens.extend(tokens)

        arr = np.array(all_tokens, dtype=np.uint16)
        out_path = out_dir / f"{name}.bin"
        arr.tofile(str(out_path))
        print(f"  {name}: {len(arr):,} tokens -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["openwebtext", "wikitext"], default="wikitext")
    parser.add_argument("--out-dir", type=str, default="data/wikitext")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "openwebtext":
        prepare_openwebtext(out_dir)
    elif args.dataset == "wikitext":
        prepare_wikitext(out_dir)


if __name__ == "__main__":
    main()
