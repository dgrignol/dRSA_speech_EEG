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
from scipy.io import wavfile as scipy_wavfile

if "MKL_NUM_THREADS" in os.environ:
    try:
        int(os.environ["MKL_NUM_THREADS"])
    except (ValueError, TypeError):
        del os.environ["MKL_NUM_THREADS"]
from transformers import AutoFeatureExtractor, Wav2Vec2Model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments controlling wav2vec2 extraction."""
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
        choices=("auto", "cpu", "cuda", "mps"),
        help="Inference device. `auto` prefers MPS (Apple) or CUDA when available.",
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
    """Resolve the desired inference device, preferring accelerators when available."""
    if device_choice == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device_choice == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested MPS device, but torch.backends.mps.is_available() is False.")
        return torch.device("mps")
    if device_choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cpu")


def load_audio(audio_path: Path, target_sr: int) -> Tuple[torch.Tensor, int]:
    """Load the merged audio stimulus and resample it to the model's native rate."""
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
    except RuntimeError:
        sample_rate, waveform_np = scipy_wavfile.read(audio_path)
        if waveform_np.ndim == 1:
            waveform_np = waveform_np[:, None]
        orig_dtype = waveform_np.dtype
        waveform_np = waveform_np.astype(np.float32)
        if np.issubdtype(orig_dtype, np.integer):
            max_val = np.iinfo(orig_dtype).max
            waveform_np /= max_val
        waveform = torch.from_numpy(waveform_np.T)
    else:
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
    """Compute the product of convolution strides to derive frame-to-sample spacing."""
    stride = 1
    if hasattr(model_config, "conv_stride"):
        for value in model_config.conv_stride:
            stride *= value
    return stride


def run_inference(
    waveform: torch.Tensor,
    sample_rate: int,
    feature_extractor,
    model: Wav2Vec2Model,
    device: torch.device,
    chunk_seconds: float,
) -> List[torch.Tensor]:
    """Run wav2vec2 forward passes in manageable chunks and gather per-layer embeddings."""
    chunk_size = int(round(chunk_seconds * sample_rate))
    if chunk_size <= 0:
        raise ValueError("chunk_seconds must yield a positive chunk size.")

    layer_segments: List[List[torch.Tensor]] | None = None
    model.eval()

    with torch.inference_mode():
        num_samples = waveform.size(0)
        for start in range(0, num_samples, chunk_size):
            end = min(start + chunk_size, num_samples)
            chunk = waveform[start:end].numpy()
            inputs = feature_extractor(
                chunk,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs.input_values.to(device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            outputs = model(
                input_values,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states
            if layer_segments is None:
                layer_segments = [[] for _ in hidden_states]
            for idx, state in enumerate(hidden_states):
                layer_segments[idx].append(state[0].cpu())

    if not layer_segments:
        raise RuntimeError("No audio samples processed; check chunking arguments.")

    concatenated_layers: List[torch.Tensor] = []
    for segments in layer_segments:
        concatenated_layers.append(torch.cat(segments, dim=0))
    return concatenated_layers


def main() -> None:
    """Entry point: orchestrate feature extraction and persist outputs."""
    args = parse_args()
    device = resolve_device(args.device)

    if not args.audio.is_file():
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    args.output.mkdir(parents=True, exist_ok=True)

    # Load wav2vec2 feature extractor and model weights.
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model)
    model = Wav2Vec2Model.from_pretrained(args.model)
    model.to(device)

    target_sr = getattr(feature_extractor, "sampling_rate", None)
    if target_sr is None:
        raise ValueError(
            "Selected model does not expose a sampling_rate. "
            "Please choose a wav2vec2 checkpoint with an associated feature extractor."
        )
    waveform, sample_rate = load_audio(args.audio, target_sr)

    # Chunk the waveform through the model to generate embeddings.
    hidden_by_layer = run_inference(
        waveform=waveform,
        sample_rate=sample_rate,
        feature_extractor=feature_extractor,
        model=model,
        device=device,
        chunk_seconds=args.chunk_seconds,
    )

    # Prepare timing metadata for downstream alignment using the final layer.
    last_layer = hidden_by_layer[-1]
    embeddings_np = last_layer.numpy()
    total_stride = compute_total_stride(model.config)
    frame_stride = total_stride / float(sample_rate)
    time_axis = np.arange(embeddings_np.shape[0]) * frame_stride

    base = args.output / args.output_base

    layer_mats: List[str] = []
    layer_metadata: List[dict] = []
    try:
        from scipy.io import savemat

        num_hidden_layers = getattr(model.config, "num_hidden_layers", len(hidden_by_layer) - 1)

        layer_labels = ["feature_extractor"]
        for layer_idx in range(1, num_hidden_layers + 1):
            layer_labels.append(f"transformer_layer_{layer_idx:02d}")
        if len(layer_labels) < len(hidden_by_layer):
            layer_labels.extend(
                [
                    f"extra_layer_{idx}"
                    for idx in range(len(layer_labels), len(hidden_by_layer))
                ]
            )

        for idx, layer_tensor in enumerate(hidden_by_layer):
            layer_np = layer_tensor.numpy()
            layer_file = base.with_name(f"{base.name}_layer{idx:02d}.mat")
            savemat(
                layer_file,
                {
                    "embeddings": layer_np,
                    "time_axis": time_axis,
                    "model_name": args.model,
                    "sample_rate": sample_rate,
                    "frame_stride": frame_stride,
                    "layer_index": idx,
                    "layer_label": layer_labels[idx],
                },
            )
            layer_mats.append(layer_file.name)
            layer_metadata.append(
                {
                    "index": idx,
                    "label": layer_labels[idx],
                    "mat_file": layer_file.name,
                    "shape": list(layer_np.shape),
                }
            )
    except Exception as exc:  # pragma: no cover - optional dependency
        warn_path = base.with_suffix(".layers.failed")
        warn_path.write_text(
            f"Unable to write layer MAT files due to: {exc}\nInstall scipy to enable MAT export.\n"
        )
        raise

    # Record run metadata for reproducibility/debugging.
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
        "layers": layer_metadata,
    }

    metadata_path = base.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print("Layered wav2vec2 embeddings saved:")
    for layer_file in layer_mats:
        print(f"  - {layer_file}")
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
