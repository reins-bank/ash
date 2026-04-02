"""Merge multiple tokens.bin files into train.bin / val.bin for training.

Streams through source files in chunks so memory usage stays bounded
regardless of dataset size.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

from ash.data.storage import PipelineFS, _s3_key


CHUNK_TOKENS = 1024 * 1024  # 1M tokens per read chunk (~2 MB)


def _count_tokens_local(path: str) -> int:
    return Path(path).stat().st_size // 2  # uint16 = 2 bytes


def _count_tokens_s3(fs: PipelineFS, path: str) -> int:
    fs._ensure_s3()
    key = _s3_key(path)
    resp = fs._client.head_object(Bucket=fs._bucket, Key=key)
    return resp["ContentLength"] // 2


def merge_token_files(
    token_files: list[str],
    output_dir: str,
    val_ratio: float = 0.1,
    fs: PipelineFS | None = None,
) -> dict:
    """Merge multiple tokens.bin files into train.bin and val.bin.

    Streams through files in chunks to keep memory usage low.

    Args:
        token_files: Paths to tokens.bin files (local or S3 keys).
        output_dir: Directory to write train.bin and val.bin.
        val_ratio: Fraction of tokens for validation (default 10%).
        fs: PipelineFS for S3 support. None = local filesystem.

    Returns:
        Dict with token counts and output paths.
    """
    if fs is None or not fs.is_s3:
        return _merge_local(token_files, output_dir, val_ratio)
    return _merge_s3(token_files, output_dir, val_ratio, fs)


def _merge_local(
    token_files: list[str], output_dir: str, val_ratio: float,
) -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train_path = str(Path(output_dir) / "train.bin")
    val_path = str(Path(output_dir) / "val.bin")

    # Count total tokens
    total = sum(_count_tokens_local(f) for f in token_files)
    split_at = int(total * (1 - val_ratio))
    print(f"  Merging {len(token_files)} file(s): {total:,} tokens "
          f"({split_at:,} train / {total - split_at:,} val)")

    written = 0
    with open(train_path, "wb") as train_f, open(val_path, "wb") as val_f:
        for src in token_files:
            mm = np.memmap(src, dtype=np.uint16, mode="r")
            offset = 0
            while offset < len(mm):
                end = min(offset + CHUNK_TOKENS, len(mm))
                chunk = mm[offset:end]
                chunk_bytes = chunk.tobytes()

                if written + len(chunk) <= split_at:
                    train_f.write(chunk_bytes)
                elif written >= split_at:
                    val_f.write(chunk_bytes)
                else:
                    # Split point falls inside this chunk
                    boundary = split_at - written
                    train_f.write(chunk[:boundary].tobytes())
                    val_f.write(chunk[boundary:].tobytes())

                written += len(chunk)
                offset = end

    return {
        "total_tokens": total,
        "train_tokens": split_at,
        "val_tokens": total - split_at,
        "train_path": train_path,
        "val_path": val_path,
    }


def _merge_s3(
    token_files: list[str],
    output_dir: str,
    val_ratio: float,
    fs: PipelineFS,
) -> dict:
    fs._ensure_s3()

    train_path = fs.join(output_dir, "train.bin")
    val_path = fs.join(output_dir, "val.bin")

    # Count total tokens
    total = sum(_count_tokens_s3(fs, f) for f in token_files)
    split_at = int(total * (1 - val_ratio))
    print(f"  Merging {len(token_files)} file(s): {total:,} tokens "
          f"({split_at:,} train / {total - split_at:,} val)")

    # Stream through S3 objects and write train/val via multipart upload
    from ash.data.storage import _S3StreamWriter

    train_key = _s3_key(train_path)
    val_key = _s3_key(val_path)
    print(f"  -> s3://{fs._bucket}/{train_key}")
    print(f"  -> s3://{fs._bucket}/{val_key}")

    written = 0
    with _S3StreamWriter(fs._client, fs._bucket, train_key) as train_w, \
         _S3StreamWriter(fs._client, fs._bucket, val_key) as val_w:

        for src in token_files:
            key = _s3_key(src)
            resp = fs._client.get_object(Bucket=fs._bucket, Key=key)
            body = resp["Body"]

            buf = b""
            chunk_bytes_size = CHUNK_TOKENS * 2  # uint16 = 2 bytes
            for data in body.iter_chunks(chunk_bytes_size):
                buf += data
                while len(buf) >= chunk_bytes_size:
                    chunk_data = buf[:chunk_bytes_size]
                    buf = buf[chunk_bytes_size:]
                    n_tokens = len(chunk_data) // 2

                    if written + n_tokens <= split_at:
                        train_w.write(chunk_data)
                    elif written >= split_at:
                        val_w.write(chunk_data)
                    else:
                        boundary = (split_at - written) * 2
                        train_w.write(chunk_data[:boundary])
                        val_w.write(chunk_data[boundary:])
                    written += n_tokens

            # Flush remaining buffer
            if buf:
                n_tokens = len(buf) // 2
                if written + n_tokens <= split_at:
                    train_w.write(buf)
                elif written >= split_at:
                    val_w.write(buf)
                else:
                    boundary = (split_at - written) * 2
                    train_w.write(buf[:boundary])
                    val_w.write(buf[boundary:])
                written += n_tokens

            body.close()

    return {
        "total_tokens": total,
        "train_tokens": split_at,
        "val_tokens": total - split_at,
        "train_path": f"s3://{fs._bucket}/{train_key}",
        "val_path": f"s3://{fs._bucket}/{val_key}",
    }
