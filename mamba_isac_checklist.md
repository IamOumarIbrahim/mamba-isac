# Mamba-ISAC Implementation Checklist

Companion to `mamba_isac_briefing.tex`. Check items off as completed.

---

## Weeks 1–2: Synthetic OFDM-ISAC Channel Generator + LMMSE Validation

### Environment setup
- [ ] Create Python env (3.10+), pin versions (`requirements.txt`)
- [ ] Install PyTorch, NumPy, SciPy, `mamba-ssm`, matplotlib
- [ ] Set up repo structure: `data/`, `models/`, `baselines/`, `eval/`, `configs/`
- [ ] Fix global random seed utility for reproducibility across all scripts

### Communication channel model
- [ ] Implement Rician fading generator (configurable K-factor)
- [ ] Implement time correlation (Jakes/Clarke Doppler spectrum model)
- [ ] Parameterize: carrier frequency, subcarrier spacing, symbol duration
- [ ] Parameterize mobility → Doppler shift mapping ($\nu_c = v f_c / c$)
- [ ] Add configurable SNR / AWGN injection
- [ ] Unit test: channel autocorrelation matches theoretical Jakes model
- [ ] Unit test: K-factor=0 reduces to Rayleigh fading (sanity check)

### Sensing (target) channel model
- [ ] Implement point-target echo model (Eq. sensing_echo in briefing)
- [ ] Parameterize target range $R$, velocity $v$, reflectivity $\alpha$
- [ ] Implement steering vectors $\mathbf{a}_r(\theta)$, $\mathbf{a}_t(\theta)$
- [ ] Add multi-path/clutter toggle (off by default, scoped out for v1)
- [ ] Unit test: recovered delay/Doppler from noiseless echo matches ground truth range/velocity

### Pilot structure
- [ ] Implement shared comb-pattern ISAC pilot allocation
- [ ] Make pilot density a config parameter
- [ ] Verify pilot orthogonality / no unintended overlap between comm and sensing pilots

### Dataset generation pipeline
- [ ] Script to generate train/val/test splits with fixed seeds
- [ ] Store joint state labels $\mathbf{s}_t$ = (CSI, range, Doppler) per sample
- [ ] Add config file (YAML/JSON) for all channel parameters
- [ ] Sanity-check dataset shapes match expected tensor dimensions
- [ ] Save small "toy" dataset (few hundred samples) for fast iteration

### LMMSE baseline + validation
- [ ] Implement closed-form LMMSE estimator (per-snapshot)
- [ ] Run LMMSE on generated dataset across SNR sweep
- [ ] Compare LMMSE NMSE curve against published/theoretical LMMSE curves
- [ ] Flag and resolve any discrepancy before proceeding (generator correctness gate)
- [ ] Document generator assumptions and known limitations in `data/README.md`

**Exit criterion for this phase:** LMMSE NMSE-vs-SNR curve from your generator matches theoretical expectation within acceptable tolerance. Do not proceed to Week 3 until this passes.

---

## Weeks 3–4: Mamba-ISAC Backbone + Dual Heads

### Backbone implementation
- [ ] Implement dual-domain input embedding (frequency-domain CSI + delay-Doppler sensing tokens)
- [ ] Integrate `mamba-ssm` selective-scan block as backbone unit
- [ ] Implement bidirectional selective scan (forward + reverse pass)
- [ ] Stack residual Mamba blocks (config: depth, hidden dim)
- [ ] Add positional/time-slot encoding if needed for sequence order

### Dual output heads
- [ ] Implement communication head (linear projection → $\hat{\mathbf{H}}_c$)
- [ ] Implement sensing head (regression → $\hat{R}, \hat{\nu}_s$)
- [ ] Decide head architecture: shared trunk depth vs. task-specific layers
- [ ] Implement joint weighted loss (Eq. loss: $\lambda_c, \lambda_s, \lambda_d$)

### Unit / sanity tests
- [ ] Toy sequence test: overfit tiny synthetic batch to near-zero loss (verifies gradient flow)
- [ ] Shape-check every tensor through forward pass
- [ ] Gradient check: no NaNs/Infs across 100 training steps
- [ ] Ablate bidirectional vs. unidirectional scan on toy data (expect bidirectional ≥ unidirectional)
- [ ] Verify $\mathcal{O}(T)$ scaling empirically: measure runtime vs. sequence length, confirm linear trend

### Training infrastructure
- [ ] Training loop with checkpointing
- [ ] Logging (loss curves, per-task loss breakdown) via TensorBoard/W&B
- [ ] Early stopping / best-checkpoint selection on validation NMSE
- [ ] Config-driven hyperparameters (learning rate, batch size, $\lambda$ weights)

**Exit criterion for this phase:** Model overfits toy dataset cleanly; full training run on real synthetic dataset converges without instability.

---

## Weeks 5–6: Transformer Baseline + LMMSE Baseline Finalization

### Transformer baseline
- [ ] Implement encoder-only self-attention architecture
- [ ] Match parameter count to Mamba-ISAC (± small tolerance) for fair comparison
- [ ] Reuse same dual-domain input embedding as Mamba-ISAC (isolate architecture as the only variable)
- [ ] Reuse same dual output heads and joint loss
- [ ] Train Transformer baseline with same training infrastructure/config discipline
- [ ] Verify Transformer overfits toy dataset (same sanity check as Mamba-ISAC)

### LMMSE baseline finalization
- [ ] Extend LMMSE (Week 1–2 version) to also output sensing range/Doppler via matched filtering or periodogram method
- [ ] Confirm LMMSE has no learned components (pure closed-form, no unfair advantage/disadvantage from training)

### Fairness controls
- [ ] Confirm all three methods (LMMSE, Transformer, Mamba-ISAC) evaluated on identical test set
- [ ] Confirm identical noise realizations across methods where applicable (paired comparison)
- [ ] Log parameter count, FLOPs estimate for each method now (before full eval phase)

**Exit criterion for this phase:** All three estimators implemented, trained (where applicable), and produce sane outputs on a held-out toy batch.

---

## Weeks 7–8: Full Metric Suite + Ablations

### Core metric suite
- [ ] NMSE computation for communication channel (all 3 methods)
- [ ] RMSE computation for range estimate (all 3 methods)
- [ ] RMSE computation for Doppler estimate (all 3 methods)
- [ ] FLOPs measurement (e.g. `thop`/`fvcore` or manual count)
- [ ] Parameter count table
- [ ] Wall-clock inference latency (mean ± std over N runs, same hardware)
- [ ] Populate Table "Planned Comparison Table" in briefing doc with real numbers

### Ablation 1: sequence-length scaling
- [ ] Run all 3 methods across pilot-history lengths (e.g. 4, 8, 16, 32, 64 slots)
- [ ] Plot NMSE vs. sequence length
- [ ] Plot latency vs. sequence length (expect Mamba-ISAC sub-quadratic vs. Transformer)

### Ablation 2: mobility (Doppler) sweep
- [ ] Run all 3 methods across Doppler/velocity range (low → high mobility)
- [ ] Plot accuracy degradation curve per method

### Ablation 3: pilot density sweep
- [ ] Run all 3 methods across pilot densities (sparse → dense)
- [ ] Plot accuracy vs. overhead trade-off curve

### Ablation 4: SNR sweep
- [ ] Run all 3 methods across SNR range (matched across methods)
- [ ] Plot NMSE/RMSE vs. SNR

### Loss weight tuning
- [ ] Grid or random search over $\lambda_c, \lambda_s, \lambda_d$
- [ ] Select final weights via validation set (not test set)
- [ ] Document final chosen weights and justification

### Statistical rigor
- [ ] Multiple random seeds per config (≥3) for mean ± std reporting
- [ ] Confirm no test-set leakage into hyperparameter selection

**Exit criterion for this phase:** All tables/plots for the paper have real numbers, all ablations complete, results are reproducible from saved configs/seeds.

---

## Weeks 9–10: Analysis, Writing, Figures, Paper Draft

### Analysis
- [ ] Identify headline result (main NMSE/latency win) for abstract framing
- [ ] Identify any negative/mixed results — do not omit, address honestly in limitations
- [ ] Sanity-check every reported number traces back to a saved log/checkpoint

### Figures
- [ ] Architecture diagram (replace placeholder in briefing `.tex`)
- [ ] NMSE vs. SNR plot (all 3 methods)
- [ ] Latency vs. sequence length plot (all 3 methods)
- [ ] Mobility sweep plot
- [ ] Pilot density sweep plot
- [ ] Export all figures as vector (PDF) at consistent font size/style

### Writing
- [ ] Update Abstract with real headline numbers (remove "proposed"/prospective framing)
- [ ] Update Introduction contributions list with confirmed results
- [ ] Fill in Results/Evaluation section with all tables populated
- [ ] Write Discussion: what results mean, where Mamba-ISAC wins/loses, why
- [ ] Write honest Limitations paragraph (single-target, synthetic-only, OFDM-only — from briefing scope)
- [ ] Update Conclusion (no new information, 3-part structure per IEEE convention)
- [ ] Verify every equation numbered and referenced in text
- [ ] Verify every table uses `\toprule`/`\midrule`/`\bottomrule`
- [ ] Verify no undefined abbreviations on first use
- [ ] Verify every claim has a citation or a number backing it

### Final checks before submission
- [ ] Full LaTeX compile (pdflatex → bibtex → pdflatex → pdflatex) with zero errors
- [ ] Proofread for economy (cut filler sentences)
- [ ] Check target venue page limit / formatting compliance
- [ ] Confirm all code/data reproducibility artifacts ready to release (repo, configs, seeds)

**Exit criterion for this phase:** Fully compilable paper draft, real results throughout, ready for internal review pass.
