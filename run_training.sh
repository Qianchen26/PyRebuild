#!/bin/bash
# Train seeds 0-4 in parallel; a seed whose 20% health check raises a red flag (exit 3) goes to asha.py.
# Usage: bash run_training.sh [seeds...]    Logs: logs/train_seedK.log, logs/asha_seedK.log
cd "$(dirname "$0")"
mkdir -p logs
export OMP_NUM_THREADS=1
for k in ${@:-0 1 2 3 4}; do
    (
      python -W ignore train.py --seed "$k" > "logs/train_seed$k.log" 2>&1
      code=$?
      if [ $code -eq 3 ]; then
        echo "seed $k: red flag at 20%, running ASHA" >> "logs/train_seed$k.log"
        python -W ignore asha.py --seed "$k" > "logs/asha_seed$k.log" 2>&1
        echo "seed $k: asha exit $?" >> "logs/train_seed$k.log"
      fi
      echo "seed $k: done (train exit $code)" >> "logs/train_seed$k.log"
    ) &
done
wait
