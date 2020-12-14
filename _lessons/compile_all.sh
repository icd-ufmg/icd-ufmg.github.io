#!/bin/bash

set -e
set -u

for f in *.ipynb; do
  bash compile_notebook.sh $f;
done
