# OFDM-ISAC Synthetic Channel Generator Documentation

## Overview

The synthetic channel generator models a 6G monostatic OFDM Integrated Sensing and Communication (ISAC) link, parameterized according to Section III of `mamba_isac_briefing.tex`.

## Channel Models

### 1. Communication Channel (`data/comm_channel.py`)
- **Fading**: Rician fading model with configurable $K$-factor ($K_{\text{dB}} = 10 \log_{10} K_{\text{linear}}$).
- **Time Correlation**: Jakes/Clarke Doppler power spectrum ($R_h(\Delta t) = J_0(2\pi f_d \Delta t)$), generated via Cholesky decomposition of the autocorrelation matrix.
- **Doppler Shift**: $\nu_c = \frac{v f_c}{c}$.
- **Noise Model**: Complex Additive White Gaussian Noise (AWGN) parameterized by SNR in dB.

### 2. Sensing (Target) Channel (`data/sensing_channel.py`)
- **Echo Formulation**: Point-target reflection model:
  $$y_s[k,t] = \alpha e^{-j 2\pi k \Delta f \tau} e^{j 2\pi t T_s \nu_s} \mathbf{a}_r(\theta) \mathbf{a}_t^T(\theta) x[k,t]$$
- **Target Parameters**: Range $R$ (delay $\tau = 2R/c$), radial velocity $v$ (Doppler $\nu_s = 2v f_c / c$), complex reflectivity $\alpha$, azimuth angle $\theta$.

### 3. Pilot Allocation (`data/pilots.py`)
- **Pattern**: Comb-type pilot arrangement across subcarriers (default spacing 4 subcarriers).
- **Orthogonality**: Shared pilot subcarrier mask between communication estimation and sensing illumination.

## Baseline Validation & LMMSE Gate (`eval/validate_lmmse.py`)
- Evaluated closed-form per-snapshot LMMSE estimator across an SNR sweep (-5 dB to 30 dB).
- Verified that empirical NMSE decreases monotonically with SNR and tracks theoretical expectations.

## Known Assumptions & Scope Limitations (v1)
1. Single monostatic base station, single communication user, single point-like sensing target.
2. Clutter/multi-path clutter toggled off by default for v1 benchmark.
3. Uniform Linear Array (ULA) with half-wavelength element spacing.
