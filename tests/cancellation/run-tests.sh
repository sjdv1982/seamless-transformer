#!/usr/bin/env bash
# One pytest process per file (repo convention: combining files in one run yields
# spurious cross-file failures). In-process files first (fast, no services), then
# the remote multi-tenant files (start a local cluster; slow).
set -u

INPROC="test_inprocess_membership_set.py test_inprocess_hard_cancel.py test_inprocess_atomicity.py"
REMOTE="test_remote_multitenant_jobserver.py test_remote_multitenant_dask.py"

for i in $INPROC $REMOTE; do
    echo "=== $i ==="
    pytest -s -v "$i"
    echo "DONE $i"
done
