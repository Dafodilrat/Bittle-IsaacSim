# Start from Isaac Sim 4.5.0 base image
FROM nvcr.io/nvidia/isaac-sim:4.5.0

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
ENV ISAACSIM_PATH=/isaac-sim

# Clone the repo
WORKDIR ${ISAACSIM_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libx11-xcb1 \
    libxcb1 \
    libxcb-render0 \
    libxcb-shm0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-sync1 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0 \
    libglu1-mesa \
    python3 python3-pip \
    libnss3 libatk-bridge2.0-0 libxcomposite1 \
    libxcursor1 libxi6 libxrandr2 libxss1 libxtst6 

RUN apt-get install -y --no-install-recommends pkg-config\
    libglvnd-dev\ 
    libgl1-mesa-dev\
    libegl1-mesa-dev\ 
    libgles2-mesa-dev
    #libgles2-mesa-dev:i386 

COPY 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN apt install -y libgl1 libnvidia-gl-570 libnvidia-common-570 mesa-utils

RUN pip3 install PyQt5 vtk
RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/Dafodilrat/Bittle-IsaacSim.git "$ISAACSIM_PATH/alpha"

RUN ${ISAACSIM_PATH}/python.sh -m pip install --no-cache-dir \
    gymnasium \
    stable-baselines3 \
    numpy \
    scipy \
    matplotlib

# Add startup script
COPY docker_entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Run via entrypoint (handles git pull + python startup)
ENTRYPOINT ["/bin/bash"]