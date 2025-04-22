#!/bin/bash
# This script is used to run the docker container for the RANO 2.0 pipeline.

INPUT_DIR=/home/aaron/KCL_data/RANO/Input_Data  # Change this to your input data directory
REPORTS_DIR=/home/slicer/Projects/Reports  # Change this to your reports directory

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
-v "${INPUT_DIR}":/home/aaron/KCL_data/RANO/Input_Data \
-v "${REPORTS_DIR}":/home/researcher/rano2.0-assist/RANO/Resources/Reports \
--user "${UID}:${GID}" \
slicerrano "${@}"

# -v "${INPUT_DIR}":/home/researcher/rano2.0-assist/input_data \

#  `$(UID)` and `$(GID)` are the user ID and group ID of the current user, respectively. This ensures that the files created
# inside the container are owned by the current user on the host machine. You can find your UID and GID by running
# `id -u` and `id -g` in the terminal.
