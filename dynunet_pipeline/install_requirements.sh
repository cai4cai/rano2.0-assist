#!/bin/bash
pip install torch torchvision  # https://pytorch.org/get-started/locally/

pip install -r requirements.txt

# add ANTs dir to path
root_dir=$(dirname "$(readlink -f "$BASH_SOURCE")")
export PATH=$root_dir/tools/ants/:$PATH
