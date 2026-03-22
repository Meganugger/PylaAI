# PylaAI

PylaAI is a Windows-only Python automation project for Brawl Stars development and experimentation. This repository is source-first: the recommended setup path is a local virtual environment plus an editable install from the repo root.

## Supported Platform

- Windows 10/11
- Python 3.10, 3.11, or 3.12
- Recommended: Python 3.11 x64

## Install

Do not use `python setup.py install`.

Use a virtual environment and install one ONNX Runtime backend explicitly.
Git must be available during install because the project uses the intended `scrcpy-client` source tag directly.

### 1. Clone the repository

```powershell
git clone https://github.com/PylaAI/PylaAI.git
cd PylaAI
```

### 2. Create and activate a virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. Install PylaAI

Recommended for most Windows users:

```powershell
python -m pip install -e ".[directml]"
```

CPU-only fallback:

```powershell
python -m pip install -e ".[cpu]"
```

NVIDIA CUDA backend:

```powershell
python -m pip install -e ".[cuda]"
```

Optional helper script:

```powershell
.\scripts\install_windows.ps1 -Backend directml -Editable
```

If you prefer a different interpreter:

```powershell
.\scripts\install_windows.ps1 -Backend cpu -Editable -PythonExecutable .\.venv\Scripts\python.exe
```

## Run

Start your emulator first, then launch the bot from the repository root:

```powershell
python main.py
```

## What This Fixes

The install flow now declares the runtime dependencies that were previously missing or only installed through fragile `setup.py` side effects, including:

- `bettercam`
- `google-play-scraper`
- `easyocr`
- `scrcpy-client` from the intended `v0.5.0` source tag
- `pywin32`
- `shapely`
- the ONNX Runtime backend you choose during install

It also removes interactive install-time prompts and subprocess-based dependency installation, which were unreliable under modern `pip install -e .` workflows.

## Dependency Notes

- `scrcpy-client==0.4.7` was removed because it hard-required `adbutils<2.0.0`, which made `pip install -e ".[cpu|directml|cuda]"` fail once the project also pinned `adbutils==2.12.0`.
- The project now depends on `scrcpy-client` from the intended `v0.5.0` source tag, which matches the dependency strategy the old installer was already trying to force manually.
- The explicit `adbutils==2.12.0` and `av==12.3.0` top-level pins were removed so pip can resolve the scrcpy transport stack cleanly from the intended scrcpy package metadata instead of hitting a resolver conflict.

## GPU / OCR Notes

- Runtime inference device selection still follows `cfg/general_config.toml` via `cpu_or_gpu`.
- `.[directml]` is the recommended default on Windows.
- `.[cuda]` is available for NVIDIA systems that should use the CUDA ONNX Runtime wheel.
- EasyOCR is installed automatically, but GPU OCR remains optional and is controlled separately by `easyocr_gpu` in `cfg/general_config.toml`.

## Tests

```powershell
python -m unittest discover
```

## Localhost Mode

This open-source repository runs in localhost mode by default. Cloud login, remote API services, and auto-update behavior are disabled unless you wire in your own backend.

## Links

- [Discord](https://discord.gg/xUusk3fw4A)
- [Trello](https://trello.com/b/SAz9J6AA/public-pyla-trello)

## License

Please respect the included license terms. This project is not permitted to be sold or monetized.
