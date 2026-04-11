for i in test_*.py; do
    echo $i
    pytest -s $i
    echo DONE $i
done

for i in cmd/test_*.py persistent/test_*.py; do
    echo $i
    pytest -s $i
    echo DONE $i
done
