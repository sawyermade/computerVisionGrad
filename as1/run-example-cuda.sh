#!/bin/bash
conda activate cv
python3 smcImgCuda.py images/school.png output/school-10-8-7-40-3-3.png 10 8 7 40 3 3 false 0
conda deactivate
eog output/building-out-2000-8-7-20-2-2-exp1.png &
