# Sparse Decomposition of Bilinear MLP Weights

My solution to the **sparse-decomposition exercise** from
[`tdooms/bilinear-decomposition`](https://github.com/tdooms/bilinear-decomposition) —
the official code repository for Pearce, Dooms, Rigg, Oramas & Sharkey,
*"Bilinear MLPs enable weight-based mechanistic interpretability"*
([arXiv:2410.08417](https://arxiv.org/abs/2410.08417)).

The exercise ships a skeleton notebook and a `Sparse` CPD model. My contribution is the
end-to-end implementation, the six-prior experiment suite, and the analysis.

**→ Full writeup: [`exercises/report.md`](exercises/report.md)**

## What this does

A single-layer bilinear MNIST classifier (test accuracy 0.968) has its computation fully
described by a symmetric interaction tensor `B ∈ ℝ^{10×784×784}`, where
`logit_c(x) = Σ_ij B[c,i,j] x_i x_j`. Eigendecomposing each class slice forces orthogonality
and re-derives shared strokes per class. Instead, I jointly factorise `B` with a CPD,

```
B[c,i,j] ≈ Σ_r L[i,r] R[j,r] D[c,r],
```

so every neuron is shared across classes through `D`, then test six structural priors to
find which one yields neurons that are simultaneously faithful, class-specialised, and
interpretable — and, just as importantly, which interpretability metric to trust.

## Results at a glance

Under a strict reconstruction rule (cosine similarity ≥ 0.99):

| Decomposition | similarity | accuracy | specialization |
|---|---:|---:|---:|
| Baseline CPD (rank 64, no prior) | 0.9982 | 0.9679 | 0.238 |
| **L1 on `D` (λ=10)** | **0.9949** | **0.9674** | **0.586** |

- **Positive result.** L1 on the class-participation factor `D` roughly 2.5× the per-neuron
  class specialization (0.238 → 0.586) for a 0.0005 accuracy cost.
- **Negative result (the more interesting one).** Input-sparsity, symmetry, non-negativity,
  and rank constraints all leave the headline specialization *flat*, even though each visibly
  moves its own target metric. Conclusion: **`D`-sparsity measures class binding, not feature
  meaningfulness — and the two are dissociable.**
- **Gauge diagnostic.** A scale-gauge stress test shows the L1 penalty starts gaming the CPD
  scale invariance past λ=10 (raw `‖D‖₁` flatlines while `‖D‖₁/(‖L‖·‖R‖)` rises), which is
  exactly where specialization saturates.

See [`exercises/report.md`](exercises/report.md) for the full derivation, all six experiments,
the seed sweep, and the mechanistic explanations.

## Build & run

The upstream repo uses `uv sync`. On the hardware I ran this on, the pinned CUDA wheels needed
a conda environment, so I added `setup.sh`:

```bash
bash setup.sh                              # conda env + pinned torch/torchvision (CUDA 12.6)
conda activate bilinear-decomposition
cd exercises
jupyter nbconvert --to notebook --execute 0_decomposition.ipynb \
    --output 0_decomposition.executed.ipynb --ExecutePreprocessor.timeout=1800
```

All figures and metric files are written under `exercises/results/` (MNIST auto-downloads on
first run). Reproducibility notes — seeds, system requirements, determinism — are in
[`exercises/report.md` §11](exercises/report.md).

## Built on

This repository is a fork of [`tdooms/bilinear-decomposition`](https://github.com/tdooms/bilinear-decomposition).
The `src/` package, the `tutorials/`, the exercise skeleton, and the `Sparse` model are
**upstream**. My contribution lives in **`exercises/`** (the implemented notebook, all of
`results/`, and `report.md`) plus the conda `setup.sh`. Full breakdown in
[`ATTRIBUTION.md`](ATTRIBUTION.md).
