# wav2vec2 Feature Extraction Overview

This note explains how `B2_model_wav2vec2.py` generates wav2vec2 embeddings for `data/Stimuli/Audio/audio_original_merged.wav`, what artefacts it writes to `Models/wav2vec2`, and the main assumptions or limitations baked into the workflow.

## Processing Workflow

- **Environment & device selection**  
  The script expects to run inside the repository’s `.venv` virtual environment. Up front it sanitises the `MKL_NUM_THREADS` variable (removing invalid values) and then selects an inference device using the `--device` CLI flag (`auto` by default). `auto` prefers Apple MPS on Apple Silicon, otherwise CUDA when available, and finally CPU.

- **Model & feature extractor**  
  It downloads the Hugging Face checkpoint specified by `--model` (default `facebook/wav2vec2-large-xlsr-53`). The script uses `AutoFeatureExtractor` plus `Wav2Vec2Model`, focusing only on continuous feature extraction (no tokenizer/vocabulary is required for this model).

- **Audio loading and resampling**  
  Audio is loaded via `torchaudio.load`. If torchaudio lacks a backend capable of reading the WAV, it falls back to SciPy’s WAV reader, normalises integer PCM values into floats in [-1, 1], and converts to mono by averaging channels. The waveform is resampled to the sampling rate expected by the model (typically 16 kHz for wav2vec2 checkpoints).

- **Chunked inference**  
  The waveform is processed in `chunk_seconds` segments (default 20 s) to avoid excessive memory use. Each chunk is passed through the feature extractor and model; the resulting `last_hidden_state` tensors are concatenated to form the final embedding sequence.

- **Stride & timing metadata**  
  To map frames back to wall-clock time, the script multiplies the model’s convolution strides to derive a total stride, divides by the sample rate, and builds a `time_axis` vector in seconds.

- **Outputs**  
  Embeddings and metadata are written under the requested output directory (`Models/wav2vec2` when called from MATLAB).

## Output Files

All outputs share a base name (default `wav2vec2_embeddings`):

| File | Contents |
| --- | --- |
| `.npy` | Raw embeddings, shape `(num_frames, hidden_size)`; e.g. `(N, 1024)` for the default model. Each row corresponds to one wav2vec2 frame. |
| `.mat` | MATLAB-friendly struct written via `scipy.io.savemat` containing: <br>• `embeddings`: same array as the `.npy` file.<br>• `time_axis`: vector of frame-onset times in seconds.<br>• `model_name`: checkpoint identifier string.<br>• `sample_rate`: resampled audio sampling rate used for inference.<br>• `frame_stride`: seconds per frame (`time_axis` step size). |
| `.json` | Human-readable metadata including absolute paths, selected model, device, chunk length, frame/hidden dimensions, frame stride, and a sparsified `time_axis_seconds` (every ~1000th value to keep the file compact). |
| `.mat.failed` (optional) | Created only if SciPy cannot write the MATLAB file; contains the failure reason so the run results are still accessible. |

## Assumptions & Limitations

- **Sampling rate alignment**: The script resamples the audio to the model’s configured sampling rate (`feature_extractor.sampling_rate`). For `facebook/wav2vec2-large-xlsr-53` this is 16 kHz. Any original sampling rate is therefore normalised to the model’s requirement.
- **Chunk granularity**: `chunk_seconds` controls memory usage. Very long audio files may benefit from reducing this value; extremely small chunks can incur additional overhead.
- **WAV input only**: The fallback loader relies on `scipy.io.wavfile.read`, so non-WAV formats still require `torchaudio` (with the relevant codecs) to succeed.
- **GPU acceleration**: On Apple Silicon, MPS is used when available; otherwise execution falls back to CPU. CUDA is supported but depends on the environment where the script runs.
- **Model compatibility**: The code assumes the chosen checkpoint exposes a `sampling_rate` via its feature extractor. Custom or fine-tuned checkpoints lacking that attribute will raise an error.

Running `B2_model_wa2vec2.m` in MATLAB orchestrates all of the above: it switches to the repo root, activates the `.venv` interpreter via `pyenv`, and invokes the Python helper with the correct paths and parameters.
