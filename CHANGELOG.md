# Changelog

All notable changes to Nile-ViT will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial repo bootstrap: `pyproject.toml`, `.gitignore`, `LICENSE`, `Makefile`,
  pre-commit config, GitHub Actions CI, package skeleton, smoke test.
- Project Plan and Technical PRD in `docs/`.

[Unreleased]: https://github.com/ammarkhattab/Nile-ViT/compare/HEAD...HEAD

### Added
- M1 Prithvi-EO-2.0 smoke test notebook (`notebooks/00_prithvi_smoke_test.ipynb`)
  validated on Colab T4: backbone loads (303.9M params), forward pass produces
  expected (1, 589, 1024) tokens per layer, LoRA attaches with 1.78% trainable
  params and stays compatible with forward pass.

### Fixed
- Install cell pins `numpy>=2.0,<2.1` (numba compat) and uninstalls `torchao`
  to bypass peft 0.19's torchao version check on Colab's pre-installed 0.10.0.

### Milestones
- ✅ **M0** — Repo bootstrapped, CI green
- ✅ **M1** — TerraTorch + Prithvi-EO-2.0 smoke test passes on Colab T4
