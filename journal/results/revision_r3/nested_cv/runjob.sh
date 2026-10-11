#!/bin/bash
# one queue job, single-threaded BLAS; log per job (run from the repo root)
cd "$(dirname "$0")/../../../.."
export VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
PY=${PY:-.venv/bin/python}
tag=$(echo "$*" | tr ' /' '__' | sed 's/--//g' | cut -c1-150)
start=$(date +%s)
$PY -m journal.pipeline.revision_r3_nested "$@" > journal/results/revision_r3/nested_cv/joblogs/$tag.log 2>&1
rc=$?
echo "$(date '+%H:%M:%S') rc=$rc $(( $(date +%s)-start ))s $*" >> journal/results/revision_r3/nested_cv/joblogs/_done.txt
