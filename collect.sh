#!/bin/bash
set -euo pipefail

# Collect outputs from a StatefulSet pod (which stays Running, so cp works).
# Rank 0 == pod -0 deterministically, so that's the one to copy from.
#
# Usage:
#   ./collect.sh mnist-ddp-0
#   ./collect.sh mnist-baseline-0
#   ./collect.sh mnist-pipeline-3

POD="${1:?usage: ./collect.sh <pod>   (e.g. mnist-ddp-0 | mnist-baseline-0 | mnist-pipeline-3)}"

DEST="$HOME/DIPLOMSKI/DistributedNeuralNetworks/output/$POD"
mkdir -p "$DEST"

echo "Collecting from $POD ..."
kubectl cp "$POD:/app/runs"   "$DEST/runs"
kubectl cp "$POD:/app/models" "$DEST/models"
echo "Done -> $DEST   (runs/ and models/ inside)"