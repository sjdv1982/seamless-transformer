#!/bin/bash
set -u -e
ntrials="${1:-250}"
ndots="${2:-30000000}"
seamless-run -i calc_pi.py -vvv ./calc-pi-nested.sh "$ntrials" "$ndots"
