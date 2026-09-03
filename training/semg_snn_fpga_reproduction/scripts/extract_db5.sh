#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw_dir="${project_dir}/data/raw"
out_dir="${project_dir}/data/extracted"

mkdir -p "${out_dir}"

for subject in {1..10}; do
  archive="${raw_dir}/s${subject}.zip"
  if [[ ! -f "${archive}" ]]; then
    echo "Missing ${archive}" >&2
    exit 1
  fi
  unzip -oq "${archive}" -d "${out_dir}"
done

find "${out_dir}" -name 'S*_E1_A1.mat' -print | sort
