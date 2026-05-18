# The Yes-Man Syndrome Website

This repository contains the project website for **The Yes-Man Syndrome: Benchmarking Abstention in Embodied Robotic Agents**.

The site is a small static/Jekyll-compatible website intended for GitHub Pages. The homepage is defined in `site/index.html`, site metadata is configured in `site/_config.yml`, and styling/assets are stored under `site/assets/`.

GitHub Pages is deployed by `.github/workflows/pages.yml`, which builds Jekyll from the `site/` directory and publishes the generated `_site` artifact.
