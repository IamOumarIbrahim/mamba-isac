# Mamba-ISAC

**Official name:** Mamba-ISAC — Selective State-Space Networks for Joint Channel Estimation in 6G Integrated Sensing and Communication
**Repo/codename:** `mamba-isac`

---

## Summary

Mamba-ISAC is a research project proposing the first selective state-space (Mamba) architecture for **joint** communication-channel and target-parameter (range/Doppler) estimation in OFDM-based integrated sensing and communication (ISAC) systems. Existing ISAC channel estimators use either closed-form LMMSE (fast, no temporal modeling) or Transformer/diffusion-based deep learning (accurate, but quadratic-complexity or iterative-latency). Mamba-ISAC targets Transformer-level accuracy at linear-time complexity by exploiting the shared physical coupling between the sensing and communication channels in a single sequential model. The project benchmarks against LMMSE and a parameter-matched Transformer under a fully synthetic OFDM-ISAC channel generator, single-target, single-cell scope for v1.

Full technical spec: `mamba_isac_briefing.tex`. Execution checklist: `mamba_isac_checklist.md`. This document covers stack + how to actually get it built with AI doing most of the labor.

---

## Recommended Software Stack

| Layer | Tool | Why |
|---|---|---|
| Language | Python 3.11 | Single-language stack — keeps AI-agent context simple, no cross-framework glue |
| DL framework | PyTorch 2.x + CUDA | Required by `mamba-ssm`; standard, well-supported by coding agents |
| SSM backbone | `mamba-ssm` + `causal-conv1d` | Official Gu/Dao reference implementation — don't reimplement the scan kernel from scratch |
| Channel simulation | NumPy + SciPy (custom) | Matches the closed-form equations in the briefing doc directly; fully transparent, easy to unit-test against theoretical LMMSE curves |
| Config management | Hydra or plain YAML + `dataclasses` | Every experiment reproducible from one config file |
| Experiment tracking | Weights & Biases (or TensorBoard if offline) | Loss curves, per-task loss breakdown, ablation sweep comparison |
| Testing | `pytest` | Every checklist "unit test" item becomes a real test file |
| Profiling | `fvcore` or `thop` | FLOPs/parameter counting for the fairness table |
| Version control | Git + GitHub | Also gives Claude Code a diff-able history to reason over |
| Environment | `venv`/`conda` + `requirements.txt`; Docker optional | Reproducibility across machines |
| Paper | LaTeX (IEEEtran), local TeX Live or Overleaf | `.tex` briefing already exists as scaffold |
| Plotting | Matplotlib/Seaborn | All ablation plots in the checklist |

**Deliberately excluded for v1:** Sionna / QuaDRiGa (heavier, TensorFlow/MATLAB-based channel simulators). They're useful later for multi-target/ray-tracing extensions, but add a second framework and slow down an AI-agent-driven build for a scope this contained. Revisit if the project extends past single-target OFDM.

---

## Using AI to Do Most of the Work

The checklist (`mamba_isac_checklist.md`) is already phase-gated with explicit exit criteria — that structure is what makes AI delegation safe: each phase has a concrete pass/fail check before the agent moves on, so errors don't compound silently across ten weeks.

### Core principle
Delegate **implementation**. Keep **verification** human. The riskiest failure mode isn't broken code — it's code that runs, looks plausible, and is quietly wrong (a channel model with a sign error, a leaked test set, a fabricated citation). Every phase below has a named human checkpoint; don't skip it even when the agent reports success.

### Setup (Day 1)
1. Open the project in **Claude Code** (terminal, VS Code, JetBrains, or the desktop app — whichever you already use).
2. Give it both artifacts as context: `mamba_isac_briefing.tex` (equations, architecture spec) and `mamba_isac_checklist.md` (task list).
3. Ask it to scaffold the repo: folder structure, `requirements.txt`, config system, empty test files matching each checklist item.
4. Commit immediately. Every phase below should end in a commit.

### Weeks 1–2 — Channel generator (AI-heavy, verification-heavy)
- Point Claude Code at the exact equations in Section III of the `.tex` file — copy the LaTeX directly into the prompt, don't paraphrase from memory. Ask it to implement the Rician generator, the sensing echo model (Eq. `sensing_echo`), and the pilot structure as separate, unit-tested modules.
- Have it write the LMMSE validation script itself.
- **Human checkpoint:** open the resulting NMSE-vs-SNR plot yourself. Compare it against the theoretical LMMSE curve shape you'd expect. This is the single most important check in the whole project — every later result depends on this generator being physically correct. Do not let the agent self-certify this ("tests pass" ≠ "physics is right").

### Weeks 3–4 — Mamba-ISAC backbone
- Feed the agent the SSM update equations (Eq. `ssm_state`, `ssm_output`) and the dual-head/loss spec (Eq. `loss`) directly from the `.tex`.
- Ask for the toy-overfit sanity test as an automated gate — a script that fails loudly if the model can't drive loss near zero on 10 samples.
- **Human checkpoint:** watch the loss curve on the toy set yourself once. An agent reporting "converged" on a broken setup is a common failure mode.

### Weeks 5–6 — Baselines
- Ask for the Transformer baseline explicitly matched in parameter count to Mamba-ISAC (give it the exact param count as a target).
- Ask it to reuse the same input embedding and heads as Mamba-ISAC — the only difference should be the backbone. This isolation is what makes the comparison fair; state this constraint explicitly, don't assume the agent infers it.
- **Human checkpoint:** print and eyeball both models' parameter counts before running anything else.

### Weeks 7–8 — Metrics, ablations, loss tuning
- This phase parallelizes well: ask the agent to write one script per ablation (sequence-length, mobility, pilot density, SNR sweeps) driven by the same config system.
- Require ≥3 seeds per config from the start — tell the agent this explicitly, it's easy to silently skip.
- **Human checkpoint:** spot-check that hyperparameter tuning used validation loss, not test loss. Ask the agent to point you to the exact line where it splits data if you're unsure.

### Weeks 9–10 — Writing
- Feed real result logs/CSVs back to Claude (chat or Code) and ask it to draft the Results/Discussion sections and populate the placeholder tables in the `.tex` directly.
- Explicitly instruct: every number in the draft must trace to a saved log file — no rounding, no estimating, no "approximately."
- **Human checkpoint:** this is where fabrication risk is highest in any AI-assisted paper. Read every populated number against its source CSV yourself before compiling. Also verify no new citations were added that weren't checked against real sources — reuse the bibliography already in the briefing `.tex` rather than letting the agent invent new ones.

### What not to delegate
- Physical correctness of the channel model (verify against theory yourself, Weeks 1–2).
- Fairness of the baseline comparison (confirm matched parameter count yourself).
- Any claim that ends up in the paper's abstract or contributions list — read the source data before it's asserted as fact.
- Citations — only cite sources you've verified exist; don't let an agent add references it "recalls."

---

## Suggested Workflow Split

| Task type | Best tool |
|---|---|
| Writing/debugging the actual codebase | Claude Code |
| Orchestrating the full multi-week pipeline hands-off | Cowork |
| Drafting paper prose from finished result logs | Claude (chat), fed the CSVs directly |
| One-off equation/architecture questions mid-implementation | Claude (chat) |
