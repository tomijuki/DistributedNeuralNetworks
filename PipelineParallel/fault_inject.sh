#!/bin/bash
set -euo pipefail

# Inject a network fault into ONE pod to simulate a straggler and measure its
# effect on collective synchronization.
#
# IMPORTANT: gloo runs over TCP. Packet loss does NOT corrupt gradients (TCP
# retransmits) -- it destroys THROUGHPUT. TCP goodput ~ MSS/(RTT*sqrt(loss)),
# so even 1% loss at 100ms RTT collapses a 2.8MB all_reduce to a crawl.
# Use delay for a measurable slowdown; use loss only in tiny amounts.
#
# Usage:
#   ./fault_inject.sh on   mnist-ddp-2 mild            # apply until removed
#   ./fault_inject.sh temp mnist-ddp-2 moderate 30     # apply for 30s, auto-remove
#   ./fault_inject.sh temp mnist-ddp-2 severe 60
#   ./fault_inject.sh on   mnist-ddp-2 custom 25ms 0.1
#   ./fault_inject.sh off  mnist-ddp-2
#   ./fault_inject.sh show mnist-ddp-2
#
# Target a NON-logging rank:  DDP -> mnist-ddp-2,  pipeline -> mnist-pipeline-1

ACTION="${1:?usage: ./fault_inject.sh <on|temp|off|show> <pod> [level] [args]}"
POD="${2:?missing pod name}"
LEVEL="${3:-mild}"

# netem's default queue is 1000 packets; a multi-MB all_reduce burst overflows
# it and drops far more than the nominal loss rate. Raise it.
LIMIT=20000

case "$LEVEL" in
  mild)     DELAY="10ms"; LOSS="0.05" ;;
  moderate) DELAY="30ms"; LOSS="0.05" ;;
  severe)   DELAY="50ms"; LOSS="0.05" ;;
  custom)   DELAY="${4:?custom needs a delay, e.g. 25ms}"; LOSS="${5:-}" ;;
  *) echo "unknown level '$LEVEL' (mild|moderate|severe|custom)" >&2; exit 1 ;;
esac

# For `temp`, the duration is the 4th arg (or the 6th when level=custom).
if [ "$LEVEL" = "custom" ]; then
  DURATION="${6:-30}"
else
  DURATION="${4:-30}"
fi

apply_fault() {
  # NOTE: no jitter. Jitter reorders packets within a flow, which TCP reads
  # as loss -> spurious retransmits -> exponential backoff -> apparent hang.
  local netem="delay $DELAY limit $LIMIT"
  [ -n "$LOSS" ] && netem="$netem loss ${LOSS}%"
  echo "Injecting into $POD [$LEVEL]: $netem"
  kubectl exec "$POD" -- sh -c "tc qdisc replace dev eth0 root netem $netem"
}

remove_fault() {
  kubectl exec "$POD" -- tc qdisc del dev eth0 root 2>/dev/null || true
}

case "$ACTION" in
  on)
    apply_fault
    kubectl exec "$POD" -- tc qdisc show dev eth0
    ;;

  temp)
    apply_fault
    # Remove the fault even if this script is Ctrl-C'd or killed, so a stray
    # qdisc can't silently poison the rest of the run.
    trap 'echo; echo "interrupted -- removing fault"; remove_fault; exit 130' INT TERM

    echo "Holding fault for ${DURATION}s  (t=0 $(date +%T))"
    for ((i = DURATION; i > 0; i--)); do
      printf "\r  %3ds remaining " "$i"
      sleep 1
    done
    printf "\r                     \r"

    remove_fault
    trap - INT TERM
    echo "Fault removed after ${DURATION}s ($(date +%T))"
    kubectl exec "$POD" -- tc qdisc show dev eth0
    ;;

  off)
    echo "Removing fault from $POD"
    remove_fault
    kubectl exec "$POD" -- tc qdisc show dev eth0
    ;;

  show)
    kubectl exec "$POD" -- tc qdisc show dev eth0
    ;;

  *)
    echo "unknown action '$ACTION' (on|temp|off|show)" >&2; exit 1 ;;
esac