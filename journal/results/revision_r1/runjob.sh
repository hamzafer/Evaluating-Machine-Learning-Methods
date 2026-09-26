#!/bin/bash
# one queue job, single-threaded BLAS; log per job
cd /Users/hamza/developer/thesis/Evaluating-Machine-Learning-Methods
export VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONWARNINGS=ignore
tag=$(echo "$*" | tr ' /' '__' | cut -c1-150)
start=$(date +%s)
.venv/bin/python -m journal.pipeline.revision_r1 "$@" > journal/results/revision_r1/joblogs/$tag.log 2>&1
rc=$?
echo "$(date '+%H:%M:%S') rc=$rc $(( $(date +%s)-start ))s $*" >> journal/results/revision_r1/joblogs/_done.txt
