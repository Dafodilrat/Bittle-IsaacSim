# Start from Isaac Sim 4.5.0 base image
FROM nvcr.io/nvidia/isaac-sim:4.5.0

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
ENV ISAACSIM_PATH=/isaac-sim

# Clone the repo
WORKDIR ${ISAACSIM_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/Dafodilrat/Bittle-IsaacSim.git "$ISAACSIM_PATH/alpha"

RUN ${ISAACSIM_PATH}/python.sh -m pip install --no-cache-dir \
    gymnasium \
    stable-baselines3 \
    PyQt5 \
    vtk \
    numpy \
    scipy \
    matplotlib

# Add startup script
COPY docker_entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Run via entrypoint (handles git pull + python startup)
ENTRYPOINT ["/entrypoint.sh"]
