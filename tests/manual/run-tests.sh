for i in $(ls *.py | grep -v \.jupyter.py); do
    echo $i
    python $i
    echo DONE $i
done
for i in *.jupyter.py; do
    echo $i
    ./jupyter-wrapper $i
    echo DONE $i
done
