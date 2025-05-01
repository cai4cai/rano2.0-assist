#!/bin/bash
# This script is used to run the docker container for the RANO 2.0 pipeline.

THIS_DIR=$(dirname "$(readlink -f "$0")")
RANO_DIR=$(dirname "$THIS_DIR")
TEST_DATA_DIR=${RANO_DIR}/data/test_data

INPUT_DIR=${RANO_DIR}/data/test_data  # Change this to your input data directory if needed
REPORTS_DIR=${RANO_DIR}/Reports  # Change this to your reports directory if needed

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

xhost +local:docker  # Allow docker to connect to X

docker run \
--rm -it \
-e DISPLAY="${DISPLAY}" \
--gpus all \
--ipc=host \
--device /dev/dri:/dev/dri \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v "${INPUT_DIR}":/home/researcher/rano2.0-assist/data \
-v "${REPORTS_DIR}":/home/researcher/rano2.0-assist/Reports \
-v "${TEST_DATA_DIR}":/home/researcher/rano2.0-assist/data/test_data \
--user "${UID}:${GID}" \
slicerrano "${@}"

# -v "${INPUT_DIR}":/home/researcher/rano2.0-assist/input_data \

#  `$(UID)` and `$(GID)` are the user ID and group ID of the current user, respectively. This ensures that the files created
# inside the container are owned by the current user on the host machine. You can find your UID and GID by running
# `id -u` and `id -g` in the terminal.
