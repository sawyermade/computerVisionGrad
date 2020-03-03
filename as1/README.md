# Assignment 1 Mean Shift Sementation: Python3, Numba, Cuda

## Install Conda Env:
```bash
$ conda create -n cv python=3.6 accelerate imageio tqdm cudatoolkit numba -y
``` 

## Run Cuda:
  python3 smcImgCuda.py input output steps hc hd m sdc sdd grayscale cardNumber

### Example:
```bash
$ conda activate cv
$ python3 smcImgCuda.py images/school.png output/school-10-8-7-40-3-3.png 10 8 7 40 3 3 false 0
$ conda deactivate
```

## Description:
```
Check Assignment1.pdf
```
