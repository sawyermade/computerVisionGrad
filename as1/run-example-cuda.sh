#!/bin/bash
conda activate cv
python3 smcImgCuda.py images/building.png output/building-out-2000-8-7-20-2-2-exp1.png 0 2000
conda deactivate
eog output/building-out-2000-8-7-20-2-2-exp1.png &
