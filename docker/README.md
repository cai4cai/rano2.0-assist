# docker instructions

## docker build

Run the following command from the directory above the 'slicerrano' directory to build the docker image:

    docker build -f rano2.0-assist/RANO/docker/Dockerfile -t slicerrano .

## docker run

    docker run --rm -it \
    -e DISPLAY=$DISPLAY \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \  
    -v ${INPUT_DIR}:/home/researcher/rano2.0-assist/input_data \
    -v ${REPORTS_DIR}:/home/researcher/rano2.0-assist/RANO/Resources/Reports \
    --user $(UID):$(GID) \
    slicerrano

- `INPUT_DIR` is the path to the directory containing the input data on the host machine.
- `REPORTS_DIR` is the path to the directory where the reports will be saved on the host machine. Make sure this 
directory exists before running the command and is writable by the user specified in the `--user` flag.
- `$(UID)` and `$(GID)` are the user ID and group ID of the current user, respectively. This ensures that the files created 
inside the container are owned by the current user on the host machine. You can find your UID and GID by running
`id -u` and `id -g` in the terminal.