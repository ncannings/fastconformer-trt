#!/bin/bash
# Like run_in_container.sh, for machines that are themselves the NeMo container (no Docker inside, e.g. RunPod pods):
# runs python directly. Expects the container's paths to exist as directories or symlinks: /w (this engine/ directory),
# /data (the data directory) and /plug (engine/plugins). Environment variables pass through unchanged.
cd /w && ASR_DATA=${ASR_DATA:-/data} PYTHONPATH=/w exec python "$@"
