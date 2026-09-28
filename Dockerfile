FROM isaac-lab-template:latest

WORKDIR /workspace/robot_rl
COPY source/robot_rl/ source/robot_rl/

RUN /isaac-sim/python.sh -m pip install --no-deps --no-build-isolation -e source/robot_rl

# README's optional sim2sim dependencies; huggingface_hub is in the base image.
RUN /isaac-sim/python.sh -m pip install --no-cache-dir 'mujoco>=3.3,<4' pygame
RUN /isaac-sim/python.sh -m pip install --no-cache-dir mediapy

# train_policy.py uses the Weights & Biases logger.
RUN /isaac-sim/python.sh -m pip install --no-cache-dir 'wandb>=0.19,<0.24'

# Allow the host user (added to Isaac Sim's group at runtime) to write Kit state.
RUN install -d -m 2775 -o root -g 1234 \
        /isaac-sim/kit/data \
        /isaac-sim/kit/logs \
        /isaac-sim/kit/cache/DerivedDataCache \
        /isaac-sim/kit/cache/nv_shadercache \
    && chgrp 1234 /isaac-sim/kit/cache \
    && chmod 2775 /isaac-sim/kit/cache
