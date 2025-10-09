# dRSA_speech_EEG

Analysis code for dRSA on EEG data recorded durign speech listening.

## Python Environment for wav2vec2 Features (macOS Apple Silicon)

The wav2vec2 pipeline relies on Python and PyTorch. To keep the setup reproducible and compatible with MATLAB, create and manage a repository-local virtual environment:

1. Ensure a recent Python (3.10 or 3.11) is available. Homebrew example: `brew install python@3.11`.
2. From the repository root, create and activate the virtual environment:
   ```bash
   cd /path/to/dRSA_speech_EEG
   python3.11 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```
3. Install the required packages (no CUDA wheels are needed on Apple Silicon, PyTorch will use Metal/MPS automatically):
   ```bash
   pip install -r requirements.txt
   ```
4. Freeze the environment so collaborators can recreate it:
   ```bash
   pip freeze > requirements.txt
   ```
5. Commit `requirements.txt` to version control. Do **not** commit the `.venv` directory.

### MATLAB Integration Tips

- Load the repository virtual environment once per MATLAB session with:
  ```matlab
  venvPy = fullfile(pwd, '.venv', 'bin', 'python');
  pe = pyenv;
  if pe.Status == "Loaded" && pe.Version ~= venvPy
      error("MATLAB already loaded a different Python: %s", pe.Version);
  elseif pe.Status ~= "Loaded"
      pyenv('Version', venvPy);
  end
  ```
- Run `B2_model_wa2vec2` only after the environment exists; MATLAB cannot switch Python versions mid-session.
- On macOS Metal devices, PyTorch automatically uses MPS for acceleration; avoid CUDA-specific wheels intended for NVIDIA GPUs.
