#!/bin/bash

set -e
set -u

if [ $# -lt 1 ]; then
  echo "Usage: $0 notebook.ipynb"
  exit 1
fi

jupyter nbconvert --execute --ExecutePreprocessor.enabled=True \
        --ExecutePreprocessor.timeout=600 --to notebook --inplace $1
jupyter nbconvert --execute --ExecutePreprocessor.enabled=True \
        --ExecutePreprocessor.timeout=600 --to markdown $1
