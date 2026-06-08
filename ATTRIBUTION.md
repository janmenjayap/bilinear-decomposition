# Attribution

This repository is a fork of
[`tdooms/bilinear-decomposition`](https://github.com/tdooms/bilinear-decomposition),
the official code repository for:

> M. Pearce, T. Dooms, A. Rigg, J. M. Oramas, and L. Sharkey.
> *Bilinear MLPs enable weight-based mechanistic interpretability.* 2024.
> [arXiv:2410.08417](https://arxiv.org/abs/2410.08417)

The upstream repository contains a "sparse decomposition exercise" — a skeleton notebook
plus a `Sparse` CPD model — intended to be completed by the reader. This fork is my
completed solution to that exercise.

## What is upstream (not my work)

- `src/` — the entire package (`image/`, `language/`, `sae/`, `shared/`, `toy/`), including
  `src/image/sparse.py` (the `Sparse` CPD model) and `src/image/model.py` (the bilinear model).
- `tutorials/` — upstream tutorial notebooks (I executed them, which only added output cells).
- The skeleton of `exercises/0_decomposition.ipynb` (a ~180-line starter) and the upstream
  packaging (`pyproject.toml`, `uv.lock`).

## What is my contribution

- `exercises/report.md` and `exercises/report.pdf` — the full writeup and analysis.
- `exercises/0_decomposition.ipynb` — the implemented notebook (the upstream skeleton was
  ~180 lines; the completed notebook is ~3,300), containing the CPD fit, the six structural
  priors, the metrics, the seed sweep, the gauge stress test, and all plotting.
- `exercises/results/` — every figure, CSV, and log produced by the notebook.
- `setup.sh` and `requirements.txt` — conda-based environment setup (the upstream `uv` path
  failed on my hardware's CUDA wheels).
- A 4-line fix to `src/language/transformer.py` (call `post_init()` for HF weight init).

## License

The upstream repository does not specify an explicit license. This fork adds no license of
its own and is shared for portfolio and review purposes only. All credit for the underlying
method, the bilinear architecture, and the exercise scaffold belongs to the authors above.
If you intend to reuse any of this code, please consult the upstream authors regarding terms.
