#!/usr/bin/env bash
# New Python environment and pinned OpenFold source; no inherited site-packages.
set -euo pipefail
PACKAGE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORK_DIR=${1:?usage: setup_env.sh WORK_DIR [WHEELHOUSE] [OPENFOLD_SOURCE_TAR]}
mkdir -p "$WORK_DIR"
WORK_DIR=$(cd -- "$WORK_DIR" && pwd)
if [[ -e "$WORK_DIR/venv" || -e "$WORK_DIR/openfold" ]]; then
    echo 'Environment already exists; choose a fresh work directory.' >&2
    exit 1
fi
"${PYTHON_BIN:-python3.12}" -m venv "$WORK_DIR/venv"
if [[ $# -ge 2 ]]; then
    "$WORK_DIR/venv/bin/python" -m pip install --no-index --find-links "$2" -r "$PACKAGE_DIR/requirements.txt"
else
    "$WORK_DIR/venv/bin/python" -m pip install --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128 -r "$PACKAGE_DIR/requirements.txt"
fi
if [[ $# -ge 3 ]]; then
    cp -- "$3" "$WORK_DIR/openfold.tar.gz"
else
    curl --fail --location --retry 2 --output "$WORK_DIR/openfold.tar.gz" https://codeload.github.com/aqlaboratory/openfold/tar.gz/be2ec1841f16c966c65ae0e7599ebbadc725757d
fi
SOURCE_TAR="$WORK_DIR/openfold.tar.gz" "$WORK_DIR/venv/bin/python" - <<'PY'
import hashlib,os,pathlib
p=pathlib.Path(os.environ['SOURCE_TAR'])
assert hashlib.sha256(p.read_bytes()).hexdigest()=='238b1d6ba21066e594bc6116226b00472fbab138ea85d672b85cbc43f4ce1820'
PY
mkdir "$WORK_DIR/openfold"
tar -xzf "$WORK_DIR/openfold.tar.gz" --strip-components=1 -C "$WORK_DIR/openfold"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=${MAX_JOBS:-8} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
export PYTHONNOUSERSITE=1
cd "$WORK_DIR/openfold"
"$WORK_DIR/venv/bin/python" setup.py build_ext --inplace
"$WORK_DIR/venv/bin/python" -m pip freeze > "$WORK_DIR/environment.freeze.txt"
"$WORK_DIR/venv/bin/python" -m pip check
echo "Environment ready. Set PYTHONPATH=$WORK_DIR/openfold for prediction."
