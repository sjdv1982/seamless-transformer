#!/bin/bash

set -u -e
export ntrials=${1:-10}
export ndots=${2:-100000000}

seeds=$(python -c '
import sys
import numpy as np
np.random.seed(0)
ntrials = int(sys.argv[1])
seeds = np.random.randint(0, 999999, ntrials)
print(" ".join([str(seed) for seed in seeds]))
' $ntrials
)

seeds=($seeds)
rm -f calc_pi-nested.job-*

cleanup_jobs() {
    pids=$(jobs -p)
    if [ -n "$pids" ]; then
        kill -1 $pids 2>/dev/null || true
        kill $pids 2>/dev/null || true
        kill -9 $pids 2>/dev/null || true
    fi
}
trap cleanup_jobs EXIT

seamless-init

for i in $(seq $ntrials); do
    i2=$((i-1))
    export seed="${seeds[$i2]}"
    cmd="python3 calc_pi.py --seed $seed --ndots $ndots > calc_pi-nested.job-$i"
    seamless-run -c "$cmd" &
    sleep 0.33
    echo $i
done

wait

head calc_pi-nested.job-*[0-9]

python3 -c '
import sys
import numpy as np
ntrials = int(sys.argv[1])
results = []
for n in range(ntrials):
    fname = f"calc_pi-nested.job-{n+1}"
    with open(fname) as f:
        data = f.read()
    try:
        curr_pi = float(data)
    except ValueError:
        print("Error for job {}".format(n+1))
        exit(1)
    results.append(curr_pi)

results = np.array(results)
print(results.mean(), results.std(), np.pi)    
' $ntrials
