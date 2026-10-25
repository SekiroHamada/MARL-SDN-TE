#!/bin/bash
# Minimal job queue worker: optionally waits for a PID to exit, then pops lines from the jobs
# file (shared safely between several workers via flock) and runs them from the repo root.
# Paths may contain spaces.
# usage: job_queue.sh <jobs_file (absolute path)> <wait_pid|0> <worker_name>
JOBS="$1"; WAIT="$2"; NAME="$3"
cd "$(dirname "$0")/.." || exit 1
[ "$WAIT" != "0" ] && tail --pid="$WAIT" -f /dev/null
export JOBS
while true; do
  CMD=$(flock "$JOBS.lock" bash -c 'l=$(head -n1 "$JOBS"); [ -n "$l" ] && sed -i 1d "$JOBS"; printf "%s" "$l"')
  [ -z "$CMD" ] && break
  echo "[$NAME] $(date +%H:%M:%S) START $CMD"
  if bash -c "$CMD"; then echo "[$NAME] $(date +%H:%M:%S) OK"; else echo "[$NAME] $(date +%H:%M:%S) FAILED $CMD"; fi
done
echo "[$NAME] QUEUE_EMPTY"
