#!/bin/bash
# This script is used to run the docker container for the RANO 2.0 pipeline.

INPUT_DIR=/home/aaron/KCL_data/RANO/input_data  # Change this to your input data directory
REPORTS_DIR=/home/aaron/KCL_data/RANO/Reports  # Change this to your reports directory

# Check if the input directory exists
if [ ! -d "$INPUT_DIR" ]; then
  echo "Input directory $INPUT_DIR does not exist."
  exit 1
fi
# Check if the reports directory exists
if [ ! -d "$REPORTS_DIR" ]; then
  echo "Reports directory $REPORTS_DIR does not exist."
  exit 1
fi

UID=$(id -u)
GID=$(id -g)

echo "Running docker container with the following parameters:"
echo "Input directory: $INPUT_DIR"
echo "Reports directory: $REPORTS_DIR"
echo "User ID: $UID"
echo "Group ID: $GID"

docker run \
--rm -it \
-e DISPLAY="${DISPLAY}" \
--gpus all \
--ipc=host \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v "${INPUT_DIR}":/home/researcher/rano2.0-assist/input_data \
-v "${REPORTS_DIR}":/home/researcher/rano2.0-assist/RANO/Resources/Reports \
--user "${UID}:${GID}" \
slicerrano "${@}"
