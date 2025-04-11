# @brief  use docker to run inference for whole brain parcellation model with MONAI
# @author Aaron Kujawa (aaron.kujawa@gmail.com).
# @date   March 2023.

# Image info
FROM nvcr.io/nvidia/pytorch:23.07-py3
LABEL maintainer="aaron.kujawa@kcl.ac.uk"
LABEL version="0.1"
LABEL description="Docker for whole brain parcellation with MONAI"

# Install commonly used packages
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    aptitude sudo vim zsh wget curl git net-tools \
    rxvt-unicode tmux thunar locales build-essential cmake \
    python3-pip
RUN echo "alias python=python3" >> ~/.bash_aliases

# Remove unnecessary packages
RUN DEBIAN_FRONTEND=noninteractive apt-get remove -y --purge gdm3 xserver-xephyr

# Install locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

ENV HOME=/home/fast_parc
WORKDIR /home/fast_parc
# Make it possible for all users to write into this directory. This way, the container can be run with any -u {UID}
# while created files are writable from the host
RUN chmod -R 777 /home/fast_parc

# Install MONAI (used for deep learning)
RUN python -m pip install pytorch-ignite
#RUN python -m pip install monai
RUN python -m pip install git+https://github.com/aaronkujawa/MONAI.git@medtronic
RUN python -m pip install nibabel
RUN python -m pip install tqdm
RUN git clone https://github.com/MIC-DKFZ/HD-BET
RUN python -m pip install -e HD-BET

# Copy necessary files
COPY src /home/fast_parc/src
COPY libs/ANTs/antsApplyTransforms.glnxa64 /usr/bin/antsApplyTransforms
COPY libs/ANTs/antsRegistration.glnxa64 /usr/bin/antsRegistration

# Change working directory
RUN mkdir -p /home/fast_parc/data/tasks/task_dir
WORKDIR /home/fast_parc/data/tasks/task_dir

# RUN CODE
ENTRYPOINT ["/home/fast_parc/data/tasks/task_dir/commands/infer.sh"]