#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p singularity/sif

echo "Building mf_swarm.sif..."
# Build the Singularity image
# This command requires sudo privileges
sudo singularity build singularity/sif/mf_swarm.sif singularity/def/mf_swarm.def > singularity/sif/mf_swarm.stdout 2>&1

if [ $? -eq 0 ]; then
    echo "Build successful! Image saved to singularity/sif/mf_swarm.sif"
    echo "Log saved to singularity/sif/mf_swarm.stdout"
else
    echo "Build failed. Check log at singularity/sif/mf_swarm.stdout"
    exit 1
fi
