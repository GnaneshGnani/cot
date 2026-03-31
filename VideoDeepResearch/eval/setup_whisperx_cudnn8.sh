#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-cot-whisperx-cudnn8}"
CONDA_EXE="${CONDA_EXE:-/home/ghazi/miniconda3/bin/conda}"
TMP_DIR="$(mktemp -d)"
CONSTRAINTS_FILE="${TMP_DIR}/whisperx_constraints.txt"
trap 'rm -rf "${TMP_DIR}"' EXIT

if [[ ! -x "${CONDA_EXE}" ]]; then
  echo "conda executable not found at ${CONDA_EXE}" >&2
  exit 1
fi

cat > "${CONSTRAINTS_FILE}" <<'EOF'
torch==2.6.0
torchaudio==2.6.0
whisperx==3.3.1
faster-whisper==1.1.0
ctranslate2==4.4.0
EOF

"${CONDA_EXE}" create -n "${ENV_NAME}" python=3.10 -y
"${CONDA_EXE}" run -n "${ENV_NAME}" python -m pip install --upgrade pip
"${CONDA_EXE}" run -n "${ENV_NAME}" python -m pip install \
  torch==2.6.0 \
  torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cpu \
  -c "${CONSTRAINTS_FILE}"
"${CONDA_EXE}" run -n "${ENV_NAME}" python -m pip install \
  whisperx==3.3.1 \
  faster-whisper==1.1.0 \
  ctranslate2==4.4.0 \
  matplotlib \
  requests \
  nvidia-cublas-cu12 \
  nvidia-cudnn-cu12==8.9.7.29 \
  -c "${CONSTRAINTS_FILE}"

"${CONDA_EXE}" run -n "${ENV_NAME}" python - <<'PY'
import importlib.metadata as md
for pkg in ("torch", "torchvision", "torchaudio", "whisperx", "faster-whisper", "ctranslate2", "matplotlib", "requests", "nvidia-cudnn-cu12"):
    try:
        print(pkg, md.version(pkg))
    except Exception as ex:
        print(pkg, "missing", ex)
PY

cat <<EOF

WhisperX sidecar environment created: ${ENV_NAME}

Use it by exporting:
  export WHISPERX_CONDA_ENV=${ENV_NAME}
  export WHISPERX_DEVICE=cuda:0
  export WHISPERX_COMPUTE_TYPE=float16
  export WHISPERX_AUX_DEVICE=cpu

The refiner will then run ASR in the sidecar env via:
  eval/whisperx_sidecar.py

EOF
