#!/usr/bin/env bash
# Run training for every experiments/**/config.json found.
# Usage:
#   ./run_all.sh                    # scans ./experiments
#   ./run_all.sh path/to/experiments
#   ./run_all.sh -- --extra foo=1   # pass extra args to training after "--"
#   LOG_DIR=logs ./run_all.sh       # optional: write per-config logs into LOG_DIR
#   DRY_RUN=1 ./run_all.sh          # print commands without executing

set -uo pipefail

# -------- Parse args --------
ROOT_DIR="experiments"
PASS_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --) shift; PASS_ARGS=("$@"); break ;;
    *)  ROOT_DIR="$1"; shift ;;
  esac
done

# -------- Options via env vars --------
: "${DRY_RUN:=0}"              # 1 => don't execute, just print
: "${LOG_DIR:=}"               # if set, logs are written here

# -------- Setup --------
if [[ ! -d "$ROOT_DIR" ]]; then
  echo "ERROR: '$ROOT_DIR' is not a directory." >&2
  exit 1
fi

if [[ -n "$LOG_DIR" ]]; then
  mkdir -p "$LOG_DIR"
fi

# Track failures but keep going
failures=()
total=0
ok=0

# -------- Find & run --------
# Use -print0 to handle spaces/newlines in paths safely
while IFS= read -r -d '' CFG; do
  (( total++ ))
  echo "[$total] Found config: $CFG"

  # Build command
  cmd=(uv run python src/infrastructure/training.py
       --path_to_experiment_file "$CFG"
      #  "${PASS_ARGS[@]}"
  )

  # Decide logging
  if [[ -n "$LOG_DIR" ]]; then
    # Create a filesystem-friendly log filename from the config path
    safe_name="${CFG//\//__}"
    safe_name="${safe_name// /_}"
    log_file="$LOG_DIR/${safe_name%.json}.log"
    echo "    -> Logging to: $log_file"
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "${cmd[@]} > \"$log_file\" 2>&1"
    else
      if "${cmd[@]}" >"$log_file" 2>&1; then
        (( ok++ ))
        echo "    ✔ Success"
      else
        failures+=("$CFG")
        echo "    ✖ FAILED (see $log_file)"
      fi
    fi
  else
    # No logging to files
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "${cmd[@]}"
    else
      if "${cmd[@]}"; then
        (( ok++ ))
        echo "    ✔ Success"
      else
        failures+=("$CFG")
        echo "    ✖ FAILED"
      fi
    fi
  fi

done < <(find "$ROOT_DIR" -type f -name 'config.json' -print0)

# -------- Summary --------
echo
echo "Processed: $total   OK: $ok   Failed: ${#failures[@]}"
if (( ${#failures[@]} > 0 )); then
  echo "Failed configs:"
  for f in "${failures[@]}"; do
    echo "  - $f"
  done
  exit 1
fi
