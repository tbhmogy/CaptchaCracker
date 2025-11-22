# Repository Guidelines

## Project Structure & Module Organization
`CaptchaCracker/` hosts the installable library; `core.py` defines the TensorFlow models, while `__init__.py` exposes the API. Root-level helpers include `train_model.py` for supervised training over PNG files in `data/train_numbers_only*/` (filenames must equal their digit labels) and `download_captcha.py` for inference-plus-capture workflows. Reference assets live in `assets/`, zipped datasets in `data.zip`, and canonical weights in `model/`; keep personal datasets and large binaries outside git.

## Build, Test, and Development Commands
- `python -m pip install -r requirements.txt` — install pinned TensorFlow 2.5 and Pillow versions compatible with the packaged release.
- `pip install -e .` — editable install so local changes inside `CaptchaCracker/` are immediately importable.
- `python train_model.py` — trains on the prepared dataset and writes weights to `model/weights_v2.h5`; adjust epochs or early stopping by editing the script.
- `python download_captcha.py` — downloads captchas from the configured endpoint, predicts text, and renames the files for quick inspection.

## Coding Style & Naming Conventions
Write Python that follows PEP 8: four spaces, `snake_case` functions, and `CamelCase` classes (`CreateModel`, `ApplyModel`). Keep configuration constants (image dimensions, char sets, paths) at the top of the module, use descriptive tensor names, and maintain short docstrings when adding preprocessing steps. If you alter the valid character set, update both `CreateModel` and `ApplyModel` plus any dataset naming scripts.

## Testing Guidelines
The project has no automated suite, so validate changes by running `python train_model.py` on a small batch and ensuring the loss curve trends downward. Load the resulting weights through `CaptchaCracker.ApplyModel` (or `python download_captcha.py`) and compare predictions against held-out images before submitting a PR. Capture accuracy deltas or sample outputs in the PR description whenever you modify data loading, decoding, or model topology.

## Commit & Pull Request Guidelines
Prefer one logical change per commit with short imperative subjects; the existing history often uses `<type>: <summary>` (e.g., `fix: readme`). Reference linked issues, list any commands you executed (install, train, predict), and describe regenerated artifacts so reviewers can reproduce the environment. Include screenshots or text samples proving captcha predictions for all model-impacting changes.

## Model Weights & Configuration Tips
Ship only the curated weights already under `model/`; store experimental checkpoints elsewhere and document download steps instead. Keep service endpoints or API keys outside the repo—load them via environment variables when extending `download_captcha.py`, and respect remote rate limits by leaving short sleeps in scraping loops.
