FROM nvcr.io/nvidia/pytorch:25.03-py3

# Slicer 5.8.1
RUN SLICER_URL="https://download.slicer.org/bitstream/67c51fc129825655577cfee9" && \
  curl -k -v -s -L $SLICER_URL | tar xz -C /tmp


# https://download.slicer.org/bitstream/6911b598ac7b1c95e7934427  # slicer 5.10.0

# Install GUI testing tools
RUN apt-get update && \
    apt-get install -y libglu1 libpulse-mainloop-glib0 libnss3 libasound2t64 qt5dxcb-plugin  #


# create directories etc opt usr
RUN mkdir -p /etc /opt /usr


# The following are taken from: https://gitlab.com/nvidia/container-images/opengl/-/blob/ubuntu22.04/base/Dockerfile
RUN dpkg --add-architecture i386 && \
    apt-get update && apt-get install -y --no-install-recommends \
        libxau6 libxau6:i386 \
        libxdmcp6 libxdmcp6:i386 \
        libxcb1 libxcb1:i386 \
        libxext6 libxext6:i386 \
        libx11-6 libx11-6:i386 \
        vim net-tools curl \
         xserver-xorg-video-dummy \
         libxrender1 \
         libpulse0 \
         libpulse-mainloop-glib0  \
         libnss3  \
         libxcomposite1 \
         libxcursor1 \
         libfontconfig1 \
         libxrandr2 \
         libglu1 \
         awesome \
        x11vnc && \
    rm -rf /var/lib/apt/lists/*

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
        ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
        ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,compat32,utility

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf


# Required for non-glvnd setups.
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64


# create directory:
RUN mkdir -p /tmp/runtime-root

# rename the existing group
RUN groupmod -n researcher $(getent group 1000 | cut -d: -f1)


# rename the existing user to "researcher"
RUN usermod -l researcher $(getent passwd 1000 | cut -d: -f1)

# create the home directory for the user
RUN mkdir -p /home/researcher
# change ownership of the home directory
RUN chown -R researcher:researcher /home/researcher

# move the slicer download to the user home
RUN  mkdir -p /home/researcher/slicer
RUN mv /tmp/Slicer*/ /home/researcher/slicer

# make user the owner of the slicer directory
RUN chown -R researcher:researcher /home/researcher/slicer

# empty pip constraint file that leads to problems when using pip_install in Slicer
RUN rm /etc/pip/constraint.txt
RUN touch /etc/pip/constraint.txt

RUN apt-get update && apt-get install -y --no-install-recommends xvfb


# set the user
USER researcher
RUN /home/researcher/slicer/Slicer-5.8.1-linux-amd64/bin/PythonSlicer -m pip install numpy==2.0.2
RUN /home/researcher/slicer/Slicer-5.8.1-linux-amd64/bin/PythonSlicer -m pip install scikit-image==0.24.0
RUN /home/researcher/slicer/Slicer-5.8.1-linux-amd64/bin/PythonSlicer -m pip install nibabel==5.3.2
RUN /home/researcher/slicer/Slicer-5.8.1-linux-amd64/bin/PythonSlicer -m pip install tqdm==4.67.1
RUN /home/researcher/slicer/Slicer-5.8.1-linux-amd64/bin/PythonSlicer -m pip install numba==0.60.0
RUN /home/researcher/slicer/Slicer-5.8.1-linux-amd64/bin/PythonSlicer -m pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# set home directory
ENV HOME /home/researcher

# make the home directory the working directory
WORKDIR /home/researcher


COPY --chown=researcher:researcher ./dynunet_pipeline /home/researcher/rano2.0-assist/dynunet_pipeline
COPY --chown=researcher:researcher ./RANO /home/researcher/rano2.0-assist/RANO
COPY --chown=researcher:researcher ./run_command /home/researcher/rano2.0-assist/run_command
COPY --chown=researcher:researcher ./tests /home/researcher/rano2.0-assist/tests


RUN mkdir /tmp/runtime-sliceruser
ENV XDG_RUNTIME_DIR=/tmp/runtime-sliceruser
ENV DISPLAY=:10

COPY --chown=researcher:researcher ./docker/xorg.conf .
COPY --chown=researcher:researcher ./docker/start-xorg.sh .
COPY --chown=researcher:researcher ./docker/install.sh .
RUN chmod +x ./install.sh

COPY --chown=researcher:researcher ./docker/install_dependencies.py /home/researcher/rano2.0-assist/install_dependencies.py
RUN ./install.sh "/home/researcher/slicer/Slicer-5.8.1-linux-amd64/Slicer"

# copy the slicer startup script
COPY --chown=researcher:researcher ./docker/.slicerrc.py /home/researcher/slicer/Slicer-5.8.1-linux-amd64/.slicerrc.py
# copy the Slicer.ini settings file (to enable Developer mode)
COPY --chown=researcher:researcher ./docker/Slicer.ini /home/researcher/.config/slicer.org/Slicer.ini



# add to PATH
ENV PATH="/home/researcher/slicer/Slicer-5.8.1-linux-amd64:${PATH}"
ENV PATH="/home/researcher/rano2.0-assist/dynunet_pipeline/tools/ants:${PATH}"

# initial command
CMD ["Slicer"]

