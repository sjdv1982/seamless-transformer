#!/bin/bash
# Run each manual check in its own Python process, continuing after failures.
# Requires the `seamless1` environment (`conda activate seamless1`).

cd "$(dirname "$0")" || exit 1
status=0

for test_file in $(ls *.py | grep -v \.jupyter.py | grep -v '^exc-spawn.py$' | grep -v '^exc.py$'); do
    echo "$test_file"
    python "$test_file" || status=1
    echo "DONE $test_file"
done

for test_file in *.jupyter.py; do
    [ -f "$test_file" ] || continue
    echo "$test_file"
    ./jupyter-wrapper "$test_file" || status=1
    echo "DONE $test_file"
done

exit "$status"
