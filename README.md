# PylaAI

PylaAI is a Windows-only Python automation project for Brawl Stars development and experimentation. This repository is source-first: the recommended setup path is a local virtual environment plus an editable install from the repo root.

## Branch Focus

This branch is `performance`.

Its goal is to keep Pyla responsive and efficient:
- lower CPU-side overhead
- leaner runtime update cadence
- tighter thread budgets
- cleaner fallback behavior when GPU acceleration is unavailable
- better suitability for running more than one emulator instance at once

This branch intentionally favors throughput and stability over the heavier experimental systems from `strongest-bot-full`.

## Supported Platform

- Windows 10/11
- Python 3.10 x64 only
- Recommended and tested setup: Python 3.10.0 in a virtual environment

## Direct Downloads

If you do not already have the required tools, download them here first:

- Python 3.10.0 x64 installer: [python-3.10.0-amd64.exe](https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe)
- Git for Windows installer: [Git-2.53.0.3-64-bit.exe](https://github.com/git-for-windows/git/releases/download/v2.53.0.windows.3/Git-2.53.0.3-64-bit.exe)

### Python 3.10.0 installer checklist

1. Run the Python installer.
2. Tick `Add Python 3.10 to PATH`.
3. Click `Install Now`.
4. After it finishes, close the installer and reopen PowerShell or Command Prompt before running setup.

### Git for Windows installer checklist

1. Run the Git installer.
2. Accept the agreement.
3. Keep clicking `Continue` with the default options.
4. On the last screen, uncheck `View Release Notes`.
5. Click `Finish` or `Close`.

## Install

Do not use `python setup.py install`.

The easiest Windows setup path is to double-click `setup.bat` from the repo root.

It will:
- check for Python 3.10.0
- create `.venv`
- ask whether you want `CUDA`, `DirectML`, or `CPU`
- install the selected backend
- apply a recommended backend-specific thread preset to `cfg/general_config.toml`
- create `start.bat`

Use a Python 3.10.0 virtual environment and install one ONNX Runtime backend explicitly.
Git must be available during install because the project installs the intended scrcpy client source revision directly.

### 1. Clone the repository

```powershell
git clone https://github.com/Meganugger/PylaAI.git
cd PylaAI
git checkout performance
```

### 2. Recommended one-click setup

Double-click `setup.bat`

or run:

```powershell
.\setup.bat
```

`setup.bat` also writes a sane default performance preset for the backend you choose:
- `CUDA`: balanced GPU-friendly limits
- `DirectML`: slightly higher ONNX CPU-side threading than CUDA
- `CPU`: a CPU-only preset with higher worker counts

It also remembers the selected backend in `cfg/general_config.toml` so runtime provider selection and default thread tuning can stay aligned with your install choice.

If Python 3.10.0 or Git is missing, `setup.bat` / `scripts/install_windows.ps1` now prints the direct download links above plus a short Windows installer checklist.

### 3. Manual setup
Create and activate a Python 3.10.0 virtual environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel "setuptools<81"
python -c "import sys; print(sys.version)"
```

The version printed above should start with `3.10.0`.

Install PylaAI:

Recommended for most Windows users on integrated graphics or DirectML-capable systems:

```powershell
python -m pip install -e ".[directml]"
```

NVIDIA CUDA backend:

```powershell
python -m pip install -e ".[cuda]"
```

The CUDA install path is pinned to the CUDA 12.4 runtime packages plus cuDNN 9.20 inside the virtual environment:

- `nvidia-cuda-runtime-cu12==12.4.*`
- `nvidia-cublas-cu12==12.4.*`
- `nvidia-cuda-nvrtc-cu12==12.4.*`
- `nvidia-cufft-cu12>=11,<12`
- `nvidia-cudnn-cu12==9.20.*`

This avoids accidentally pulling newer CUDA runtime packages such as 12.9/13.x-adjacent tooling. For the supported PylaAI setup, you do not need to manually copy cuDNN files into a system CUDA folder.

CPU-only fallback:

```powershell
python -m pip install -e ".[cpu]"
```

Optional helper script:

```powershell
.\scripts\install_windows.ps1 -Backend directml -Editable
```

CUDA helper script:

```powershell
.\scripts\install_windows.ps1 -Backend cuda -Editable
```

If you prefer a different Python 3.10.0 interpreter:

```powershell
.\scripts\install_windows.ps1 -Backend directml -Editable -PythonExecutable .\.venv\Scripts\python.exe
```

## Run

Start your emulator first, then launch the bot by double-clicking `start.bat`.

Manual fallback from the repository root:

```powershell
python main.py
```

## Multi-Instance

The `performance` branch now supports lightweight multi-instance launches without duplicating the whole repo.

### 1. Generate per-instance configs and launchers

Run:

```powershell
python main.py --setup-instances 2
```

Replace `2` with however many emulator instances you want.

This creates:
- `instances\1\cfg`, `instances\2\cfg`, ...
- per-instance `latest_brawler_data.json` files when a saved roster already exists
- `start_1.bat`, `start_2.bat`, ...

During setup, PylaAI asks for the emulator port for each instance and writes that value into the matching instance config.

### 2. Launch an instance

Use the generated launcher for each instance:

```powershell
start_1.bat
```

Each launcher uses:

```powershell
python main.py --instance 1 --autostart
```

That means the instance:
- uses its own `instances\<n>\cfg` directory
- uses its own saved roster file
- connects to the configured emulator port instead of grabbing the first ADB device it sees
- skips the UI for lower overhead and starts the bot directly

### 3. Open the UI for a specific instance

If you want to edit roster or settings for one instance in the Control Center first, run:

```powershell
python main.py --instance 1
```

That opens the normal UI, but bound to `instances\1\cfg` and that instance's saved roster instead of the root config.

### Multi-instance efficiency notes

- When `instance_count` is greater than `1`, the performance branch now scales CPU-side thread defaults down automatically so instances do not oversubscribe the same machine as aggressively.
- `scrcpy_max_fps = "auto"` also becomes more conservative for multi-instance runs to reduce decoder overhead.
- For the best results, keep one emulator per configured ADB port and save a roster for each instance before using `--autostart`.

## Brawler Setup

The brawler setup screen now supports two optional quality-of-life tools:

- Brawl Stars API trophy sync
- Auto-push roster generation

If you enter an official Brawl Stars API key and your player tag in the setup screen, PylaAI can import your current brawler trophies and refresh configured roster entries with live values.

You can also use `Build Auto Push` to automatically create a roster of supported owned brawlers below a target trophy value. The generated roster keeps the existing saved data shape and orders brawlers from the lowest current trophies upward so the bot pushes the lowest ones first.

This API integration is optional. If you leave the API fields empty, manual brawler configuration still works exactly as before.

For best post-match reliability, the Brawl Stars API settings are strongly recommended. Without them, unresolved match recovery falls back to lobby OCR only, which is slower and can delay replaying the next game after some matches.

### How To Get And Set Up The Brawl Stars API Key

1. Go to `https://developer.brawlstars.com` and sign in with your Supercell ID.
2. Create a new API key from the developer portal.
3. Add your current public IP address to the key whitelist.
4. Copy the generated API key.
5. In PylaAI Settings, paste that value into `Brawl Stars API Key`.

Important:

- The official Brawl Stars API key is tied to your public IP address.
- If your public IP changes later, you need to update the key on the developer portal or the API sync will stop working.

### How To Find Your Player Tag

1. Open Brawl Stars.
2. Tap your profile/avatar in the top-left corner.
3. Copy your player tag from the profile screen.
4. Paste it into `Player Tag` in PylaAI Settings.

Notes:

- You can paste the tag with `#` or without it. PylaAI normalizes it automatically.
- After filling both fields, save settings and restart the bot if it was already running.

## Runtime Sync Notes

- Verified trophy refreshes now flow through the shared runtime roster, so the Live page, Control Center queue, Brawlers page, and Match History stay aligned after results are committed.
- Match result recovery uses end-screen detection first, then lobby/API verification when needed, so showdown placements and delayed lobby returns can still resolve to the correct trophy change.
- If the Brawl Stars API fields are empty, PylaAI will still run, but lobby result verification has to rely on OCR and may take longer when a result was not detected on the end screen.

## What This Fixes

The install flow now declares the runtime dependencies that were previously missing or only installed through fragile `setup.py` side effects, including:

- `bettercam`
- `google-play-scraper`
- `easyocr`
- `scrcpy-client` from the intended upstream source revision
- `pywin32`
- the ONNX Runtime backend you choose during install
- the scrcpy transport stack that brings `adbutils` and `av`
- a setuptools runtime pin compatible with `adbutils`' current `pkg_resources` import
- pinned CUDA 12.4 runtime wheels, supporting CUDA DLL wheels, and cuDNN 9.20 runtime wheels inside the venv when you install `.[cuda]`

It also removes interactive install-time prompts and subprocess-based dependency installation, which were unreliable under modern `pip install -e .` workflows.

## Dependency Notes

- PylaAI is now packaged for Python `>=3.10,<3.11` and the recommended user path is Python `3.10.0` specifically because that is the known-good setup.
- The project depends on the scrcpy client from the intended upstream source revision instead of relying on old install-time hacks.
- `adbutils` and `av` are no longer manually pinned at the top level in PylaAI packaging. They are resolved through the scrcpy dependency chain so `pip install -e ".[cpu|directml|cuda]"` can resolve cleanly.
- `setuptools` is pinned below `81` because the current `adbutils` runtime still imports `pkg_resources` and newer setuptools emits deprecation warnings for that path.
- `shapely` was removed from packaging because the current runtime no longer imports it.
- For CUDA installs, PylaAI now pins the venv runtime libraries to CUDA `12.4.*` and cuDNN `9.20.*` so pip does not drift to newer incompatible runtime packages.
- I did not hardcode a system-wide CUDA 12.4 + cuDNN 9.20 copy step because ONNX Runtime does not require that for its supported Python package flow. The supported path is to keep those DLLs inside the virtual environment and preload them at runtime.

## GPU / OCR Notes

- Runtime inference device selection still follows `cfg/general_config.toml` via `cpu_or_gpu`.
- `.[directml]` is the recommended default on Windows.
- `.[cuda]` is available for NVIDIA systems that should use the CUDA ONNX Runtime wheel.
- cuDNN 9.x is required for modern CUDA 12.x ONNX Runtime GPU usage, and PylaAI now pins that to cuDNN `9.20.*` in the CUDA extra.
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
