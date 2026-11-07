#!/bin/bash
# Train seeds 0-4 in parallel; a seed whose 20% health check raises a red flag (exit 3) goes to asha.py.
# Exits with 1 if any seed ends without a healthy model (train.py or its asha.py exits non-zero; train.py
# exits 4 when the final health check fails), 0 otherwise.
# Usage: bash run_training.sh [seeds...]    Logs: logs/train_seedK.log, logs/asha_seedK.log
cd "$(dirname "$0")"
mkdir -p logs
export OMP_NUM_THREADS=1
pids=()
for k in ${@:-0 1 2 3 4}; do
    (
      python -W ignore train.py --seed "$k" > "logs/train_seed$k.log" 2>&1
      code=$?
      echo "seed $k: train exit $code" >> "logs/train_seed$k.log"
      if [ $code -eq 3 ]; then
        echo "seed $k: red flag at 20%, running ASHA" >> "logs/train_seed$k.log"
        python -W ignore asha.py --seed "$k" > "logs/asha_seed$k.log" 2>&1
        code=$?
        echo "seed $k: asha exit $code" >> "logs/train_seed$k.log"
      fi
      exit $code
    ) &
    pids+=($!)
done
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
exit $failed
