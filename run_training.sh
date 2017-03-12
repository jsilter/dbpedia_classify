#!/bin/bash
set -x

out_log_file="$1"

git_hash=`git rev-parse HEAD`

echo `date` >> ${out_log_file}
echo ${git_hash} >> ${out_log_file}

python keras_text_classify_pt1.py | tee --append ${out_log_file}
