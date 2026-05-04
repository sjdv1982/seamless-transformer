set -u -e

for i in $(ls *.py | grep -v \.jupyter.py | grep -v '^exc-spawn.py$' | grep -v '^exc.py$'); do
    echo $i
    python $i
    echo DONE $i
done
for i in *.jupyter.py; do
    echo $i
    ./jupyter-wrapper $i
    echo DONE $i
done
