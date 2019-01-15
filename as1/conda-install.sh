#!/bin/bash
conda create -n cv python=3 accelerate imageio tqdm cudatoolkit=7.5 -y
chmod +x run-example-cuda.sh

