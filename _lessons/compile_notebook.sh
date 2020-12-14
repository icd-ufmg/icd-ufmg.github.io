#!/bin/bash

set -e
set -u

if [ $# -lt 1 ]; then
  echo "Usage: $0 notebook.ipynb"
  exit 1
fi

folder=`dirname $1`
file=`basename $1`
fname_noext=`basename $1 .ipynb`
support_files=$folder/${fname_noext}_files

if [[ -d $support_files ]]; then
  rm -rf $support_files
fi

if [[ -d $fname_noext/$support_files ]]; then
  rm -rf $fname_noext/$support_files
fi

# jupyter nbconvert --execute --ExecutePreprocessor.enabled=True \
#         --ExecutePreprocessor.timeout=600 --to notebook --inplace $1
jupyter nbconvert --execute --ExecutePreprocessor.enabled=True \
        --ExecutePreprocessor.timeout=600 --to markdown $1

if [[ ! -d $fname_noext/$support_files ]]; then
  mkdir -p $fname_noext/$support_files
fi

if [[ -d $support_files ]]; then
  cp -r $support_files/* $fname_noext/$support_files/
fi

output=$folder/${fname_noext}.md
tail -n +2 $output > $folder/${fname_noext}.md.tmp
mv $folder/${fname_noext}.md.tmp $folder/${fname_noext}.md

sed -i 's/```python/```python\n#In: /g' $folder/${fname_noext}.md
