#!/bin/bash
# Run each Dask test file in an isolated pytest process.
# Requires the `seamless1` environment (`conda activate seamless1`).
# Usage: ./run-tests.sh [pytest arguments]
#        TEST_TIMEOUT=300 ./run-tests.sh

cd "$(dirname "$0")" || exit 1
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"

status=0
for test_file in test_*.py; do
    [ -f "$test_file" ] || continue
    echo "$test_file"
    if [ -n "${TEST_TIMEOUT:-}" ]; then
        timeout --foreground "${TEST_TIMEOUT}" python -m pytest -s "$test_file" "$@" || status=1
    else
        python -m pytest -s "$test_file" "$@" || status=1
    fi
    echo "DONE $test_file"
done

exit "$status"
