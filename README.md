# Mamba-ISAC: Selective State-Space Networks for Joint Channel Estimation in 6G Integrated Sensing and Communication

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Mamba-ISAC is a research framework proposing selective state-space (Mamba) models for **joint** communication-channel ($\mathbf{H}_c$) and target-parameter (range $R$, velocity/Doppler $\nu_s$) estimation in OFDM-based integrated sensing and communication (ISAC) systems.

## Key Features

- **Sequential Dual-Domain Embedding**: Combines frequency-domain CSI and delay-Doppler sensing tokens.
- **Selective State-Space Backbone**: $\mathcal{O}(T)$ linear time complexity in pilot sequence length versus $\mathcal{O}(T^2)$ self-attention.
- **Dual Output Heads**: Shared trunk representation driving simultaneous CSI matrix estimation and range/Doppler regression.
- **Comprehensive Baselines**: Benchmarked against closed-form LMMSE and parameter-matched Transformer models.
- **Synthetic OFDM-ISAC Channel Generator**: Time-correlated Rician fading comm channel + point-target radar echo model.

## Repo Structure

```
mamba-isac/
├── configs/             # YAML configuration files
├── data/                # Rician comm generator, point-target radar generator, pilots
├── models/              # Dual-domain embeddings, Selective Scan Mamba, heads, loss
├── baselines/           # LMMSE estimator, Transformer ISAC baseline
├── eval/                # Metric calculation, evaluation pipelines, ablations
├── utils/               # Reproducibility utilities, FLOPs/param counters
├── tests/               # Pytest suite
├── scripts/             # Plotting & paper figure generation
├── mamba_isac_briefing.tex # IEEEtran LaTeX research paper
├── mamba_isac_checklist.md # Execution checklist
└── mamba_isac_project_overview.md # Project architecture overview
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run test suite:
```bash
pytest
```

3. Generate dataset:
```bash
python generate_dataset.py --config configs/default_config.yaml
```

4. Train Mamba-ISAC model:
```bash
python train.py --config configs/default_config.yaml
```

5. Run full benchmark suite:
```bash
python eval/evaluate_all.py
```

## Citation

If you use this repository in your research, please cite `mamba_isac_briefing.tex`.

```bibtex
@article{mambaisac2026,
  title={Selective State-Space Modeling for Joint Channel Estimation in OFDM-Based Integrated Sensing and Communication: A Mamba-ISAC Framework},
  author={Mamba-ISAC Team},
  journal={Research Proposal / Working Paper},
  year={2026}
}
```
