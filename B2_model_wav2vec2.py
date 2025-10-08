#!/usr/bin/env python3
"""
Extract wav2vec2.0 embeddings for the merged audio stimulus.

This script loads a pretrained wav2vec2 model from Hugging Face, processes the
audio file, and saves the resulting hidden-state embeddings alongside metadata.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute wav2vec2 embeddings for an audio stimulus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--audio",
        required=True,
        type=Path,
        help="Path to the merged audio waveform (`audio_original_merged.wav`).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory where embeddings and metadata will be stored.",
    )
    parser.add_argument(
        "--model",
        default="facebook/wav2vec2-large-xlsr-53",
        help="Hugging Face identifier of the wav2vec2 model to use.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Inference device. Use `auto` to prefer CUDA when available.",
    )
    parser.add_argument(
        "--chunk-seconds",
        default=20.0,
        type=float,
        help="Audio chunk length (in seconds) processed per forward pass.",
    )
    parser.add_argument(
        "--output-base",
        default="wav2vec2_embeddings",
        help="Base filename (without extension) for saved embeddings.",
    )
    return parser.parse_args()


def resolve_device(device_choice: str) -> torch.device:
    if device_choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_choice)


def load_audio(audio_path: Path, target_sr: int) -> Tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.dim() != 2:
        raise RuntimeError(f"Unexpected audio tensor shape {waveform.shape}")

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
        sample_rate = target_sr

    return waveform.squeeze(0), sample_rate


def compute_total_stride(model_config) -> int:
    stride = 1
    if hasattr(model_config, "conv_stride"):
        for value in model_config.conv_stride:
            stride *= value
    return stride


def run_inference(
    waveform: torch.Tensor,
    sample_rate: int,
    processor: Wav2Vec2Processor,
    model: Wav2Vec2Model,
    device: torch.device,
    chunk_seconds: float,
) -> torch.Tensor:
    chunk_size = int(round(chunk_seconds * sample_rate))
    if chunk_size <= 0:
        raise ValueError("chunk_seconds must yield a positive chunk size.")

    segments: List[torch.Tensor] = []
    model.eval()

    with torch.inference_mode():
        num_samples = waveform.size(0)
        for start in range(0, num_samples, chunk_size):
            end = min(start + chunk_size, num_samples)
            chunk = waveform[start:end].numpy()
            inputs = processor(
                chunk,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs.input_values.to(device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            outputs = model(input_values, attention_mask=attention_mask)
            segment = outputs.last_hidden_state[0].cpu()
            segments.append(segment)

    if not segments:
        raise RuntimeError("No audio samples processed; check chunking arguments.")

    return torch.cat(segments, dim=0)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if not args.audio.is_file():
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    args.output.mkdir(parents=True, exist_ok=True)

    processor = Wav2Vec2Processor.from_pretrained(args.model)
    model = Wav2Vec2Model.from_pretrained(args.model)
    model.to(device)

    target_sr = processor.feature_extractor.sampling_rate
    waveform, sample_rate = load_audio(args.audio, target_sr)

    embeddings = run_inference(
        waveform=waveform,
        sample_rate=sample_rate,
        processor=processor,
        model=model,
        device=device,
        chunk_seconds=args.chunk_seconds,
    )

    embeddings_np = embeddings.numpy()
    total_stride = compute_total_stride(model.config)
    frame_stride = total_stride / float(sample_rate)
    time_axis = np.arange(embeddings_np.shape[0]) * frame_stride

    base = args.output / args.output_base
    npy_path = base.with_suffix(".npy")
    np.save(npy_path, embeddings_np)

    try:
        from scipy.io import savemat

        savemat(
            base.with_suffix(".mat"),
            {
                "embeddings": embeddings_np,
                "time_axis": time_axis,
                "model_name": args.model,
                "sample_rate": sample_rate,
                "frame_stride": frame_stride,
            },
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        warn_path = base.with_suffix(".mat.failed")
        warn_path.write_text(
            f"Unable to write MAT file due to: {exc}\nInstall scipy to enable MAT export.\n"
        )

    metadata = {
        "audio_path": str(args.audio.resolve()),
        "output_directory": str(args.output.resolve()),
        "model_name": args.model,
        "device": str(device),
        "chunk_seconds": args.chunk_seconds,
        "num_frames": int(embeddings_np.shape[0]),
        "hidden_size": int(embeddings_np.shape[1]),
        "frame_stride_seconds": frame_stride,
        "time_axis_seconds": time_axis.tolist()[:: max(len(time_axis) // 1000, 1)],
    }

    metadata_path = base.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"Embeddings saved to {npy_path}")
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
