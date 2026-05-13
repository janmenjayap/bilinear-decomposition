#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="bilinear-decomposition"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$PROJECT_DIR"

if command -v conda &>/dev/null; then
    CONDA_BASE=$(dirname "$(dirname "$(command -v conda)")")
elif [[ -n "${CONDA_EXE:-}" ]]; then
    CONDA_BASE=$(dirname "$(dirname "$CONDA_EXE")")
else
    echo "ERROR: conda not found in PATH and CONDA_EXE is not set." >&2
    return 1 2>/dev/null || exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "Conda env '$ENV_NAME' already exists, skipping create."
else
    conda create -n "$ENV_NAME" python=3.13 -y
fi
conda activate "$ENV_NAME"

# torch/torchvision are pinned to +cu126 in requirements.txt so pip pulls the
# CUDA-enabled wheels (PyPI only ships CPU-only torch).
pip install -r requirements.txt

# Prepend $CONDA_PREFIX/lib to LD_LIBRARY_PATH on env activate. Required because
# pip-installed wheels (e.g. optree's _C.so) link against newer libstdc++ symbols
# (GLIBCXX_3.4.31+) than the system /lib/x86_64-linux-gnu/libstdc++.so.6. The
# conda env ships a newer libstdc++.so.6, but the loader only sees it if
# LD_LIBRARY_PATH points there. Without this, `import torch` (which imports
# optree) fails with "GLIBCXX_3.4.31 not found".
ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
DEACTIVATE_DIR="$CONDA_PREFIX/etc/conda/deactivate.d"
mkdir -p "$ACTIVATE_DIR" "$DEACTIVATE_DIR"
cat > "$ACTIVATE_DIR/ld_library_path.sh" <<'EOF'
export _CONDA_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
EOF
cat > "$DEACTIVATE_DIR/ld_library_path.sh" <<'EOF'
export LD_LIBRARY_PATH="${_CONDA_OLD_LD_LIBRARY_PATH:-}"
unset _CONDA_OLD_LD_LIBRARY_PATH
EOF

# Apply to the current shell too so subsequent commands in this script work.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
pip install ipykernel
python -m ipykernel install --user --name "bilinear-decomposition" --display-name "bilinear-decomposition (Python 3.13)"
pip install -e .

# conda deactivate && bash -x setup.sh && conda activate bilinear-decomposition
