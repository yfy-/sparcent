#!/usr/bin/env bash
# One-command setup of Sparcent on Linux (x86_64, NVIDIA driver supporting CUDA 11.7):
# fetches the PISA and ciff submodules, creates the mamba environment, installs the
# colbert fork with its sparcent extra (the splade fork and the BMP Python bindings)
# and builds ciff (pisa2ciff) and the PISA tools used to build the sparse centroid index.
#
# Usage: ./setup.sh                      # environment named "sparcent"
#        SPARCENT_ENV=other ./setup.sh   # custom environment name
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${SPARCENT_ENV:-sparcent}"
JOBS="${JOBS:-$(nproc)}"
PISA_TOOLS=(parse_collection invert reorder-docids lexicon)

if command -v mamba >/dev/null 2>&1; then
    CONDA=mamba
elif command -v micromamba >/dev/null 2>&1; then
    CONDA=micromamba
elif command -v conda >/dev/null 2>&1; then
    CONDA=conda
else
    echo "error: mamba, micromamba or conda is required" >&2
    exit 1
fi

echo "==> Fetching submodules"
if ! git -C "$ROOT" submodule update --init --recursive; then
    # GitHub may refuse anonymous HTTPS clones from some hosts.
    echo "==> Retrying the submodules over SSH"
    git -C "$ROOT" -c url."git@github.com:".insteadOf=https://github.com/ \
        submodule update --init --recursive
fi

echo "==> Creating the '$ENV_NAME' environment with $CONDA"
if $CONDA env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    $CONDA env update -n "$ENV_NAME" -f "$ROOT/environment.yml"
else
    $CONDA env create -n "$ENV_NAME" -f "$ROOT/environment.yml"
fi

if [ "$CONDA" != micromamba ] && command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
else
    eval "$($CONDA shell hook --shell bash)"
    $CONDA activate "$ENV_NAME"
fi

echo "==> Building ciff (pisa2ciff)"
cargo install --locked --path "$ROOT/ciff" --root "$CONDA_PREFIX"

echo "==> Building the PISA tools"
cmake -S "$ROOT/pisa" -B "$ROOT/build/pisa" -DCMAKE_BUILD_TYPE=Release \
    -DPISA_ENABLE_TESTING=OFF -DPISA_ENABLE_BENCHMARKING=OFF
cmake --build "$ROOT/build/pisa" --parallel "$JOBS" --target "${PISA_TOOLS[@]}"
for tool in "${PISA_TOOLS[@]}"; do
    ln -sf "$ROOT/build/pisa/bin/$tool" "$CONDA_PREFIX/bin/$tool"
done

echo "==> Writing the environment activation hook"
# ColBERT compiles its C++ extensions at runtime. Python.h includes crypt.h, which is in
# $CONDA_PREFIX/include, and the environment's compiler doesn't search there by default.
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/sparcent.sh" <<'EOF'
export SPARCENT_OLD_CPATH="${CPATH-}"
export CPATH="${CPATH:+$CPATH:}$CONDA_PREFIX/include"
EOF
cat > "$CONDA_PREFIX/etc/conda/deactivate.d/sparcent.sh" <<'EOF'
export CPATH="$SPARCENT_OLD_CPATH"
[ -n "$CPATH" ] || unset CPATH
unset SPARCENT_OLD_CPATH
EOF

echo "==> Checking the installation"
python -c "import colbert, splade.sparcent, bmp"
for tool in "${PISA_TOOLS[@]}" pisa2ciff nvcc; do
    command -v "$tool" >/dev/null || { echo "error: $tool not found" >&2; exit 1; }
done

echo "Done. Activate the environment with: $CONDA activate $ENV_NAME"
