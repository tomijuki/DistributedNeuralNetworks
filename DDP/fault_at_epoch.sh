#!/bin/bash
set -uo pipefail

# Wait until training reaches a given epoch, then inject a timed network fault.
#
# Usage: ./fault_at_epoch.sh <epoch> <seconds> [level] [watch-pod] [target-pod]
#   ./fault_at_epoch.sh 5 60
#   ./fault_at_epoch.sh 5 60 moderate mnist-pipeline-3 mnist-pipeline-1

EPOCH="${1:?usage: ./fault_at_epoch.sh <epoch> <seconds> [level] [watch-pod] [target-pod]}"
DURATION="${2:-60}"
LEVEL="${3:-moderate}"
WATCH_POD="${4:-mnist-ddp-0}"
TARGET_POD="${5:-mnist-ddp-2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- preflight: fail loudly instead of silently ---------------------------
if ! kubectl get pod "$WATCH_POD" >/dev/null 2>&1; then
    echo "ERROR: watch pod '$WATCH_POD' not found. Existing pods:" >&2
    kubectl get pods >&2
    exit 1
fi
if ! kubectl get pod "$TARGET_POD" >/dev/null 2>&1; then
    echo "ERROR: target pod '$TARGET_POD' not found." >&2
    exit 1
fi

echo "Watching $WATCH_POD for epoch $EPOCH ..."
echo "  will fault $TARGET_POD [$LEVEL] for ${DURATION}s"
echo "----- live log -----"

# Loose match: "Epoch 5", "Epoch  5", "Epoch 5/15" all hit; "Epoch 15" does not.
PATTERN="Epoch[[:space:]]+${EPOCH}([^0-9]|$)"

# Stream to a temp file in the BACKGROUND and poll it. Piping into `grep -m1`
# deadlocks once the pod goes quiet: grep exits, but tee/kubectl block forever
# on a read that never returns, so no SIGPIPE is ever delivered.
TMPLOG="$(mktemp)"
kubectl logs -f --tail=-1 "$WATCH_POD" > "$TMPLOG" 2>&1 &
LOG_PID=$!

HIT=""
while :; do
    HIT="$(grep -m1 -E "$PATTERN" "$TMPLOG" 2>/dev/null)"
    [ -n "$HIT" ] && break
    # stop if the log stream died (pod gone, or training already over)
    if ! kill -0 "$LOG_PID" 2>/dev/null; then
        break
    fi
    sleep 2
done

kill "$LOG_PID" 2>/dev/null
wait "$LOG_PID" 2>/dev/null

echo "----- end of watch -----"

if [ -n "$HIT" ]; then
    echo "MATCHED: $HIT"
    echo "Epoch $EPOCH reached at $(date +%T) -- injecting."
    "$SCRIPT_DIR/fault_inject.sh" temp "$TARGET_POD" "$LEVEL" "$DURATION"
else
    echo "No match for epoch $EPOCH." >&2
    echo "Check the format of the epoch line:" >&2
    echo "  kubectl logs $WATCH_POD | grep -n Epoch" >&2
    exit 1
fi