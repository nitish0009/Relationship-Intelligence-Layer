#!/usr/bin/env bash
# Fetch the ORIGINAL CARE-GNN and PC-GNN code at the exact commits used in the paper.
# Not vendored: PC-GNN ships without a licence file, so we never redistribute either repo.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party
fetch() {  # url dir commit
  if [ ! -d "third_party/$2" ]; then git clone -q "$1" "third_party/$2"; fi
  git -C "third_party/$2" fetch -q --depth 1 origin "$3" 2>/dev/null || true
  git -C "third_party/$2" checkout -q "$3"
}
fetch https://github.com/YingtongDou/CARE-GNN.git CARE-GNN a64ff7523e187a24251f7ca88435d2c9d8f7dcd9
fetch https://github.com/PonderLY/PC-GNN.git      PC-GNN   9d7d7fae491081178b6e12e193fb89cfc330f2be
echo "third_party/ ready"
