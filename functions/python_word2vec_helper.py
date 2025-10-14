"""
Lightweight utilities for exposing pretrained word2vec embeddings to MATLAB.

The helper downloads (if necessary) the Google News word2vec binary and
extracts vectors only for requested tokens to minimise the amount of data
loaded into memory.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import requests
from tqdm import tqdm
import argparse
import json
import sys
from huggingface_hub import hf_hub_download
try:  # huggingface_hub<0.23 does not expose HfHubHTTPError at top level
    from huggingface_hub.utils._errors import HfHubHTTPError  # type: ignore
except ImportError:  # pragma: no cover
    class HfHubHTTPError(Exception):
        """Fallback error when huggingface_hub does not expose HfHubHTTPError."""
        pass

MODEL_URL = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
MODEL_FILENAME = "GoogleNews-vectors-negative300.bin"
MODEL_ARCHIVE = MODEL_FILENAME + ".gz"
VECTOR_DIMENSION = 300
HUGGINGFACE_REPO = "word2vec/word2vec-google-news-300"


class Word2VecDownloadError(RuntimeError):
    """Raised when the pretrained model cannot be downloaded."""


def _ensure_model(cache_dir: Path, manual_path: Optional[str] = None) -> Path:
    """
    Ensure the decompressed word2vec binary exists within cache_dir.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    if manual_path:
        manual_path = Path(manual_path).expanduser()
        if manual_path.exists():
            return manual_path
        logging.getLogger(__name__).warning(
            "Provided manual model path %s does not exist. Falling back to cache.",
            manual_path,
        )
    env_path = os.getenv("WORD2VEC_GOOGLE_NEWS_PATH")
    if env_path:
        manual_path = Path(env_path).expanduser()
        if manual_path.exists():
            return manual_path
        logging.getLogger(__name__).warning(
            "WORD2VEC_GOOGLE_NEWS_PATH=%s does not exist. Falling back to cache.",
            manual_path,
        )

    binary_path = cache_dir / MODEL_FILENAME
    if binary_path.exists():
        return binary_path

    logging.getLogger(__name__).info(
        "Downloading word2vec model from Hugging Face to %s", binary_path
    )

    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    try:
        downloaded_path = hf_hub_download(
            repo_id=HUGGINGFACE_REPO,
            filename=MODEL_FILENAME,
            cache_dir=str(cache_dir),
            token=token,
        )
    except HfHubHTTPError as exc:
        raise Word2VecDownloadError(
            "Failed to download word2vec model. "
            "If the repository is gated, set HUGGINGFACEHUB_API_TOKEN "
            "with a valid Hugging Face token or download the binary manually."
        ) from exc

    downloaded_path = Path(downloaded_path)
    if downloaded_path != binary_path:
        try:
            os.replace(downloaded_path, binary_path)
        except OSError:
            # If another process already moved it, keep the existing file.
            if not binary_path.exists():
                raise

    if not binary_path.exists():
        raise Word2VecDownloadError(f"Unable to locate downloaded model at {binary_path}")

    return binary_path


def _iter_word2vec_binary(binary_path: Path):
    """
    Yield (token, vector) pairs from the word2vec binary file.
    """
    with open(binary_path, "rb") as f:
        header = f.readline()
        try:
            vocab_size, vector_size = map(int, header.split())
        except ValueError as exc:
            raise RuntimeError(
                f"Malformed header in word2vec binary: {header!r}"
            ) from exc

        if vector_size != VECTOR_DIMENSION:
            logging.getLogger(__name__).warning(
                "Unexpected vector size %d (expected %d).",
                vector_size,
                VECTOR_DIMENSION,
            )

        binary_len = np.dtype(np.float32).itemsize * vector_size

        for _ in range(vocab_size):
            token_bytes = []
            while True:
                ch = f.read(1)
                if not ch:
                    break
                if ch == b" ":
                    break
                if ch != b"\n":
                    token_bytes.append(ch)

            token = b"".join(token_bytes).decode("latin-1")
            if not token:
                # Skip malformed entries
                f.seek(binary_len, os.SEEK_CUR)
                leftover = f.read(1)
                if not leftover:
                    break
                continue

            vector = np.frombuffer(f.read(binary_len), dtype=np.float32)
            leftover = f.read(1)  # consume newline or separator
            if len(vector) != vector_size:
                break
            yield token, vector


def fetch_embeddings(
    tokens: Iterable[str],
    cache_dir: Optional[str] = None,
    manual_path: Optional[str] = None,
) -> Dict[str, Optional[list]]:
    """
    Return embeddings for the requested tokens as Python lists.

    Parameters
    ----------
    tokens
        Iterable of tokens to retrieve.
    cache_dir
        Optional directory to store the downloaded binary. Defaults to
        ~/.cache/dRSA_word2vec.

    Returns
    -------
    dict
        Mapping of tokens to embedding lists. Missing tokens map to ``None``.
    """
    cache_path = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "dRSA_word2vec"
    binary_path = _ensure_model(cache_path, manual_path=manual_path)

    requested = {}
    for token in tokens:
        if not token:
            continue
        key = str(token)
        if key not in requested:
            requested[key] = None
    if not requested:
        return {}

    remaining = set(requested.keys())
    for word, vector in _iter_word2vec_binary(binary_path):
        if word in remaining:
            requested[word] = vector.tolist()
            remaining.remove(word)
            if not remaining:
                break

    return requested


def _parse_args(argv: Optional[Iterable[str]] = None):
    parser = argparse.ArgumentParser(description="Fetch word2vec embeddings for given tokens.")
    parser.add_argument("--tokens", required=True, help="Path to a newline-delimited token list.")
    parser.add_argument("--output", required=True, help="Where to write the resulting JSON mapping.")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional directory to cache the word2vec binary (defaults to ~/.cache/dRSA_word2vec).",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional path to an existing GoogleNews-vectors-negative300.bin file.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)

    with open(args.tokens, "r", encoding="utf-8") as f_in:
        tokens = [line.strip() for line in f_in if line.strip()]

    embeddings = fetch_embeddings(tokens, cache_dir=args.cache_dir, manual_path=args.model_path)

    with open(args.output, "w", encoding="utf-8") as f_out:
        json.dump(embeddings, f_out)

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
