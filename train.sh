#!/bin/bash

# Set environment variables
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Number of GPUs per node
GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')

# Default values for distributed training
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6001}

# Path to the configuration file
CONFIG_PATH="config/template_args.yaml"

# Distributed training arguments
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# Run the training script
torchrun $DISTRIBUTED_ARGS train.py --config_path $CONFIG_PATH
