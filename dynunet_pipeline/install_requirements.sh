#!/bin/bash
pip install

pip install -r requirements.txt

# add ANTs dir to path
root_dir=$(dirname "$(readlink -f "$BASH_SOURCE")")
export PATH=$root_dir/tools/ants/:$PATH
