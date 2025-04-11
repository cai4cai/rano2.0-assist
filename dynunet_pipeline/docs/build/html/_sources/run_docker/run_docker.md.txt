# Run with docker

Docker allows to run the inference without installing the dependencies locally. The docker image contains all 
dependencies. The docker image can be run on a local machine or on a server. In addition, a task directory needs to be
mounted into the docker container. This directory is found under `data/tasks` in the repository. The task directory
contains the input images and the model files. The output is written by the docker container into the task directory 
as well.

## Obtain the docker image
There are two ways to obtain the docker image. The first way is to download the image from docker hub. The second way
is to build the image locally. The second way is recommended if you want to modify the code. 

### Download the docker image from docker hub
The docker image is available on docker hub. The image can be downloaded with the following command:

```docker pull aaronkujawa/fast_parcellation:latest```

This can take a while. The image is more than 20 GB in size.

### Build the docker image locally
The docker image can be build locally with the following command:

```docker build -f docker/Dockerfile -t aaronkujawa/fast_parcellation .``` (don't forget the dot at the end)

This command has to be executed in the root directory (dynunet_pipeline) of the repository.

## Run inference with docker (example Task 2120)
Inference with docker can be run with the command assembled in 
`data/tasks/task2120_regnobetprimix/commands/infer_with_docker.py`

Images to run inference on have to be placed in the task directory under
`data/tasks/task2120_regnobetprimix/input/dataset/imagesTs_fold0` directory.

An alternative input directory and other arguments can be modified in the arguments file:
`data/tasks/task2120_regnobetprimix/config/infer_args.json`


