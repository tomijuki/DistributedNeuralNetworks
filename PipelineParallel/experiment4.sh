#!/bin/bash
set -uo pipefail

# PIPELINE with fault injection on a middle stage. Run N times, collecting
# results per run and tearing down cleanly in between.
#
# Per iteration:
#   apply -> wait ready -> arm fault at epoch E -> wait for "training done"
#   -> collect to output/run-N -> delete -> wait for pods to fully terminate
#
# Usage:
#   ./experiment4.sh                       # 5 runs, DDP, mild @ epoch 10, 60s
#   ./experiment4.sh 3                     # 3 runs
#   RUNS=5 MANIFEST=pipeline-statefulset.yaml WATCH_POD=mnist-pipeline-3 \
#     TARGET_POD=mnist-pipeline-1 LABEL=app=mnist-pipeline ./experiment4.sh

RUNS="${1:-${RUNS:-5}}"

MANIFEST="${MANIFEST:-pipeline.yaml}"
LABEL="${LABEL:-app=mnist-pipeline}"
WATCH_POD="${WATCH_POD:-mnist-pipeline-3}"     # LAST stage: prints epochs + writes runs/models
TARGET_POD="${TARGET_POD:-mnist-pipeline-1}"   # middle stage: gets the fault
FAULT_EPOCH="${FAULT_EPOCH:-10}" #TODO change this back to 10
FAULT_SECS="${FAULT_SECS:-60}"
FAULT_LEVEL="${FAULT_LEVEL:-mild}"

DEST_ROOT="${DEST_ROOT:-$HOME/DIPLOMSKI/DistributedNeuralNetworks/output-pipeline-fault}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Marker main.py prints after training, before `sleep infinity`.
DONE_MARKER="training done"
READY_TIMEOUT="600s"
TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-3600}"   # seconds to wait for training to finish

echo "=========================================================="
echo " $RUNS runs | $MANIFEST"
echo " fault: $FAULT_LEVEL on $TARGET_POD @ epoch $FAULT_EPOCH for ${FAULT_SECS}s"
echo " results -> $DEST_ROOT/run-N"
echo "=========================================================="

# kubectl wait fails instantly with NotFound if the pod object doesn't exist
# yet -- the StatefulSet controller needs a moment after apply to create it.
# So: wait for EXISTENCE first, then for readiness.
wait_for_pod_exists() {
    local pod="$1" timeout="${2:-180}"
    local deadline=$(( SECONDS + timeout ))
    while ! kubectl get pod "$pod" >/dev/null 2>&1; do
        if [ "$SECONDS" -ge "$deadline" ]; then
            echo "  pod $pod never appeared after ${timeout}s" >&2
            return 1
        fi
        sleep 2
    done
    return 0
}

teardown() {
    kubectl delete -f "$MANIFEST" --ignore-not-found >/dev/null 2>&1
    kubectl delete pvc -l "$LABEL" --ignore-not-found >/dev/null 2>&1
    # Wait for the pods to actually disappear -- applying while old pods are
    # still Terminating gives you a half-old, half-new StatefulSet.
    echo -n "  waiting for teardown "
    for _ in $(seq 1 120); do
        n="$(kubectl get pods -l "$LABEL" --no-headers 2>/dev/null | wc -l)"
        [ "$n" -eq 0 ] && { echo "-> clean"; return 0; }
        echo -n "."
        sleep 2
    done
    echo " (timeout; continuing anyway)"
}

# Start from a clean slate.
teardown

for i in $(seq 1 "$RUNS"); do
    echo
    echo "########## RUN $i / $RUNS  ($(date +%T)) ##########"
    DEST="$DEST_ROOT/run-$i"
    mkdir -p "$DEST"

    kubectl apply -f "$MANIFEST" >/dev/null || { echo "apply failed"; exit 1; }

    echo "  waiting for $WATCH_POD to appear ..."
    if ! wait_for_pod_exists "$WATCH_POD" 180; then
        kubectl get pods -l "$LABEL" >&2
        teardown
        continue
    fi

    echo "  waiting for $WATCH_POD to be ready ..."
    if ! kubectl wait --for=condition=ready "pod/$WATCH_POD" --timeout="$READY_TIMEOUT"; then
        echo "  pod never became ready; skipping run $i" >&2
        kubectl get pods -l "$LABEL" >&2
        teardown
        continue
    fi

    # Arm the fault in the background; it watches the log for the epoch line.
    echo "  arming fault @ epoch $FAULT_EPOCH ..."
    "$SCRIPT_DIR/fault_at_epoch.sh" "$FAULT_EPOCH" "$FAULT_SECS" "$FAULT_LEVEL" \
        "$WATCH_POD" "$TARGET_POD" > "$DEST/fault.log" 2>&1 &
    FAULT_PID=$!

    # Block until training finishes (marker line), streaming the log to file.
    echo "  training ... (following $WATCH_POD)"
    # Stream the log to a file in the BACKGROUND and poll it for the marker.
    # (Do NOT pipe into `grep -m1`: after grep exits, `tee`/`kubectl` block on
    # a read that never returns, because the pod goes silent in `sleep
    # infinity` -- so no SIGPIPE is ever delivered and the pipeline hangs.)
    : > "$DEST/train.log"
    kubectl logs -f --tail=-1 "$WATCH_POD" > "$DEST/train.log" 2>&1 &
    LOG_PID=$!
    tail -f "$DEST/train.log" &      # live view for you
    TAIL_PID=$!

    deadline=$(( SECONDS + TRAIN_TIMEOUT ))
    while :; do
        if grep -q "$DONE_MARKER" "$DEST/train.log" 2>/dev/null; then
            break
        fi
        if [ "$SECONDS" -ge "$deadline" ]; then
            echo "  TIMEOUT after ${TRAIN_TIMEOUT}s waiting for '$DONE_MARKER'" >&2
            break
        fi
        # bail out early if the pod died instead of finishing
        phase="$(kubectl get pod "$WATCH_POD" -o jsonpath='{.status.phase}' 2>/dev/null)"
        if [ -n "$phase" ] && [ "$phase" != "Running" ] && [ "$phase" != "Pending" ]; then
            echo "  pod $WATCH_POD is $phase -- stopping wait" >&2
            break
        fi
        sleep 5
    done

    kill "$TAIL_PID" "$LOG_PID" 2>/dev/null
    wait "$TAIL_PID" "$LOG_PID" 2>/dev/null

    echo "  training finished at $(date +%T)"

    # Fault should be long done; kill the watcher if it's somehow still alive,
    # and make sure no qdisc is left behind on the target.
    kill "$FAULT_PID" 2>/dev/null
    wait "$FAULT_PID" 2>/dev/null
    "$SCRIPT_DIR/fault_inject.sh" off "$TARGET_POD" >/dev/null 2>&1

    # Collect while the pod is still up (sleep infinity keeps it alive).
    echo "  collecting -> $DEST"
    kubectl cp "$WATCH_POD:/app/runs"   "$DEST/runs"   2>/dev/null || echo "    (no runs/)"
    kubectl cp "$WATCH_POD:/app/models" "$DEST/models" 2>/dev/null || echo "    (no models/)"

    teardown
done

echo
echo "=========================================================="
echo " all $RUNS runs complete -> $DEST_ROOT"
echo " TensorBoard:  tensorboard --logdir $DEST_ROOT"
echo "=========================================================="