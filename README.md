# Explainable AI Tutorial Codes

Overview
--------
This repository contains example scripts and notebooks demonstrating Explainable AI (xAI) techniques using LIME and SHAP, plus several small utilities for embedding analysis and visualization.

Quick list of notable files
- `lime_tutorial.py`, `lime_tutorial_all_solutions.py`
- `shap_tutorial.py`, `shap_tutorial_all_solutions.py`
- `llm_embedding_analyzer.py`, `llm_embedding_probing.py`, `llm_embedding_visualizer.py`
- `requirements.txt` (Python dependencies)
- `aclImdb/` (local IMDB dataset folder used by some examples)
- `lime_explanation_*.html` (example output files)

Requirements
------------
- Python 3.8+ (3.10 or later recommended)
- Git (optional, to clone the repo)

Installation
------------
1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install pip requirements:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- `torch` is optional in `requirements.txt` — install a platform-specific wheel if you need PyTorch GPU support.
- If you are on macOS with Apple Silicon, you may prefer `tensorflow-macos` / `tensorflow-metal` instead of the generic `tensorflow` package.

Conda (alternative, recommended for reproducible binaries)
-----------------------------------------------------
Conda provides an environment manager and binary package distribution which simplifies installing compiled packages (TensorFlow, PyTorch) and system libraries.

Which distribution
- Miniconda: lightweight and recommended — https://docs.conda.io/en/latest/miniconda.html
- Anaconda: full data-science distribution (larger install) — https://www.anaconda.com/products/distribution

Download the macOS installer matching your CPU:
- Apple Silicon (M1/M2/arm64): `Miniconda3-latest-MacOSX-arm64.sh`
- Intel (x86_64): `Miniconda3-latest-MacOSX-x86_64.sh`

Install Miniconda (example):

```bash
# Apple Silicon (arm64)
curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh

# Intel (x86_64)
curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
```

After installation enable conda in your shell and restart the terminal:

```bash
conda init zsh
exec $SHELL
conda --version
```

Create and activate a conda environment for this repo (Python 3.10 recommended):

```bash
conda create -n xai_env python=3.10 -y
conda activate xai_env

# Use pip to install the repository requirements
pip install --upgrade pip
pip install -r requirements.txt
```

Conda tips
- To install PyTorch with conda (CPU-only example):

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

- Use `conda install -c conda-forge <package>` for many scientific packages when preferred.

Apple Silicon (macOS) TensorFlow notes
- For best TensorFlow support on M1/M2, follow Apple's instructions. Recommended steps:

```bash
conda install -c apple tensorflow-deps
pip install tensorflow-macos
pip install tensorflow-metal
```

- On Intel macs, `pip install tensorflow` from `requirements.txt` is usually fine for CPU usage.

Reproducible environment: example `environment.yml`
-------------------------------------------------
If you prefer a conda YAML spec, use an `environment.yml` like the example below. It creates a conda env and then installs pip requirements.

```yaml
name: xai_env
channels:
	- conda-forge
dependencies:
	- python=3.10
	- pip
	- pip:
		- -r requirements.txt
```

Create from the file:

```bash
conda env create -f environment.yml
conda activate xai_env
```

Verification & troubleshooting
- Confirm the active interpreter and pip are from the environment:

```bash
which python
python --version
which pip
pip --version
```

- If imports fail, ensure the environment is activated (`conda activate xai_env` or `source .venv/bin/activate`) and reinstall requirements.

If you'd like, I can add a real `environment.yml` file to the repo or create a condensed quickstart section. 

How to run
----------
Run any script with Python from the repository root. Example commands:

```bash
# Run the LIME tutorial
python lime_tutorial.py

# Run all LIME solutions (produces HTML explanations)
python lime_tutorial_all_solutions.py

# Run the SHAP tutorial
python shap_tutorial.py

# Run all SHAP solutions
python shap_tutorial_all_solutions.py

# Embedding utilities
python llm_embedding_analyzer.py
python llm_embedding_probing.py
python llm_embedding_visualizer.py
```

If a script accepts command-line arguments, pass `--help` to list options (for example `python lime_tutorial.py --help`).

Data
----
This repo includes an `aclImdb/` folder with IMDB samples. If any script requires more complete IMDB data, place it under the `aclImdb/` folder or update script paths accordingly.

Outputs
-------
Some scripts produce HTML result files (for example files beginning with `lime_explanation_`). Open those with your browser to view explanations.

Troubleshooting
---------------
- If you see Keras / TensorFlow compatibility errors, use the pinned versions in `requirements.txt` or create a fresh venv and reinstall.
- If an import fails, confirm you're using the virtualenv and that `pip install -r requirements.txt` completed without errors.

Next steps
----------
- Run `python lime_tutorial.py` to reproduce a quick LIME example.
- Tell me if you want me to run any script here and I can run it and share the output.

License / Attribution
---------------------
This repository is a collection of tutorial scripts. Check source files for any additional attribution or license notes.
