# Docker instructions

## Requirements
[RANO](../RANO)
#### [1. nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
The NVIDIA Container Toolkit is a collection of libraries and utilities enabling users to build and run GPU-accelerated containers.

After installing nvidia-container-toolkit, restart the Docker daemon, for example use:

    sudo systemctl restart docker

#### [2. Manage Docker as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/)
Alternatively, modify the following docker scripts such that they run docker with `sudo`.


## docker build
Make sure the docker_build.sh script can be executed:

    chmod +x docker_build.sh

Run the docker_build.sh script.

    ./docker_build.sh

## docker run

Modify the paths at the beginning of the `docker_run.sh` script.

    INPUT_DIR=...  # path to the directory containing the input data on the host machine.
    REPORTS_DIR=...  # is the path to the directory where the reports will be saved on the host machine. Make sure this directory exists before running the command and is writable by the user specified in the `--user` flag.

Make sure the docker_run.sh script can be executed:

    chmod +x docker_run.sh

Run the docker_runs.sh script.

    ./docker_run.sh