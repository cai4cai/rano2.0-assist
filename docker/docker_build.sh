#!/bin/bash

# change to the build context directory
THIS_FILE_PATH=$(readlink -f "$0")
THIS_DIR=$(dirname "$THIS_FILE_PATH")
# context dir
CONTEXT_DIR="$THIS_DIR/.."

echo CONTEX_DIR: "$CONTEXT_DIR"
cd "$CONTEXT_DIR"

docker build -f docker/Dockerfile -t slicerrano . "${@}"