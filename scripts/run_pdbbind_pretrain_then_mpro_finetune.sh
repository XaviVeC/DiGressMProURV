#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/run_pdbbind_pretrain_then_mpro_finetune.sh <PDBBIND_DIR> <MPRO_DIR>"
  echo "Example: bash scripts/run_pdbbind_pretrain_then_mpro_finetune.sh C:/data/PDBbindv2020 C:/data/MPro-URV_Version2"
  exit 1
fi

PDBBIND_DIR="${1//\\//}"
MPRO_DIR="${2//\\//}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

if [[ ! -f "${PDBBIND_DIR}/Total.csv" ]]; then
  echo "Error: PDBbind folder does not look valid: ${PDBBIND_DIR}"
  echo "Expected file not found: ${PDBBIND_DIR}/Total.csv"
  exit 1
fi

if [[ ! -f "${MPRO_DIR}/train.csv" ]]; then
  echo "Error: MPro folder does not look valid: ${MPRO_DIR}"
  echo "Expected file not found: ${MPRO_DIR}/train.csv"
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
PRETRAIN_NAME="pdbbind_pretrain_${STAMP}"
FINETUNE_NAME="mpro_finetune_${STAMP}"

echo "[1/2] Starting PDBbind pretraining..."
python src/main.py +experiment=pdbbind_pretrain \
  general.name="${PRETRAIN_NAME}" \
  dataset.raw_data_dir="${PDBBIND_DIR}"

PRETRAIN_RUN_DIR="$(find ../outputs -mindepth 2 -maxdepth 2 -type d -name "*-${PRETRAIN_NAME}" | LC_ALL=C sort | tail -n 1)"

if [[ -z "${PRETRAIN_RUN_DIR}" ]]; then
  echo "Error: could not locate the pretraining output directory for ${PRETRAIN_NAME}"
  exit 1
fi

PRETRAIN_CKPT_REPO_REL="${PRETRAIN_RUN_DIR}/checkpoints/${PRETRAIN_NAME}/last.ckpt"

if [[ ! -f "${PRETRAIN_CKPT_REPO_REL}" ]]; then
  echo "Error: pretraining checkpoint not found: ${PRETRAIN_CKPT_REPO_REL}"
  exit 1
fi

# main.py resolves general.resume relative to src/, not the repo root.
# The pretraining checkpoint is found as ../outputs/... from the repo root,
# so the resume argument must become ../../outputs/... from src/.
PRETRAIN_CKPT_FOR_RESUME="../${PRETRAIN_CKPT_REPO_REL}"

echo "[2/2] Starting MPro fine-tuning..."
python src/main.py +experiment=mpro_finetune \
  general.name="${FINETUNE_NAME}" \
  dataset.raw_data_dir="${MPRO_DIR}" \
  general.resume="${PRETRAIN_CKPT_FOR_RESUME}"

FINETUNE_CKPT_REPO_REL="${PRETRAIN_RUN_DIR}/checkpoints/${FINETUNE_NAME}_resume/last.ckpt"

echo
echo "Pretraining run directory: ${PRETRAIN_RUN_DIR}"
echo "Pretraining checkpoint:    ${PRETRAIN_CKPT_REPO_REL}"

if [[ -f "${FINETUNE_CKPT_REPO_REL}" ]]; then
  echo "Fine-tuning checkpoint:    ${FINETUNE_CKPT_REPO_REL}"
else
  echo "Fine-tuning checkpoint should appear at:"
  echo "  ${FINETUNE_CKPT_REPO_REL}"
fi

echo
echo "Done."
echo "Note: in the current DiGress resume flow, fine-tuning outputs are written inside the pretraining run directory."