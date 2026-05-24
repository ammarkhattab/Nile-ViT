# Nile-ViT

> A parameter-efficient multimodal Vision Transformer for detecting **compound** heat–drought–vegetation-stress events over the Nile Delta and Eastern Mediterranean.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Status:** 🚧 Pre-Phase-1 — repository bootstrapped, implementation in progress.

---

## What this is

Nile-ViT fuses (a) Sentinel-2 / HLS satellite imagery encoded by a LoRA-fine-tuned [Prithvi-EO-2.0](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M) geospatial foundation model with (b) co-located ERA5 + CHIRPS meteorological time series, via cross-attention, to detect *compound* climate-extreme events (drought ∧ heatwave ∧ vegetation stress) at the pixel level.

The contribution is twofold:

1. **A new public benchmark** — `nile-compound-2017-2024`, the first labeled compound-extreme dataset for the Nile Delta / Eastern Mediterranean.
2. **A parameter-efficient multimodal architecture** trained on consumer hardware (a single GTX 1650 plus free Colab T4) using LoRA on Prithvi-EO-2.0.

See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) for the strategic plan and [`docs/TECHNICAL_DESIGN.md`](docs/TECHNICAL_DESIGN.md) for the build spec.

---

## Quickstart

Requires **Python 3.11** and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/ammarkhattab/Nile-ViT.git
cd Nile-ViT
uv sync --extra dev
uv run pytest -q
```

A successful smoke test confirms the toolchain is wired up. Heavier capabilities (data download, model training, demo) require credentials for NASA Earthdata, Copernicus CDS, Hugging Face, and W&B — see [`docs/INSTALL.md`](docs/INSTALL.md) (TODO).

---

## Project layout

```
Nile-ViT/
├── nilevit/         # importable Python package
├── configs/         # TerraTorch / experiment YAMLs
├── scripts/         # data download, training, evaluation entry points
├── notebooks/       # exploratory + paper figures
├── tests/           # unit + smoke tests
├── docs/            # project plan, design doc, data card, model card
├── demo/            # Gradio app (HF Space)
└── paper/           # LaTeX source
```

---

## Roadmap

| Milestone | What | Status |
|---|---|---|
| M0 | Repo bootstrapped, CI green | 🚧 in progress |
| M1 | TerraTorch + Prithvi-EO-2.0 smoke test on Colab | ⏳ pending |
| M2–M5 | Data pipeline + HF dataset release | ⏳ pending |
| M6 | Baselines table | ⏳ pending |
| M7 | Nile-ViT v0 trained | ⏳ pending |
| M11 | Public Gradio demo | ⏳ pending |
| M13 | CCAI workshop submission | ⏳ pending |

Full milestone list with acceptance criteria: [`docs/TECHNICAL_DESIGN.md §13`](docs/TECHNICAL_DESIGN.md).

---

## Citation

If you use Nile-ViT or its dataset in your work, please cite:

```bibtex
@misc{khattab2026nilevit,
  title  = {Nile-ViT: A Parameter-Efficient Multimodal Vision Transformer for
            Detecting Compound Heat-Drought-Vegetation-Stress Events over the
            Nile Delta and Eastern Mediterranean},
  author = {Khattab, Ammar},
  year   = {2026},
  url    = {https://github.com/ammarkhattab/Nile-ViT}
}
```

---

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
