# RoboAbstention

This repository contains the code and project website for **The Yes-Man Syndrome: Benchmarking Abstention in Embodied Robotic Agents**.

## Repository Layout

- `src/` contains the benchmark generation and evaluation code.
- `site/` contains the static/Jekyll-compatible project website.
- `.github/workflows/pages.yml` deploys the website to GitHub Pages.

## Code

Python dependencies for the benchmark code are listed in `src/requirements.txt`.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r src/requirements.txt
```

## Website

The website source lives in `site/`:

- `site/index.html`
- `site/_config.yml`
- `site/assets/`

GitHub Pages is configured to build from `site/` and publish the generated `_site` artifact. The workflow must stay at the repository root under `.github/workflows/`; workflows inside `site/.github/` are not discovered by GitHub Actions.
