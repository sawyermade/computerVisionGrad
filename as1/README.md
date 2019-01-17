# Assignment 1 Mean Shift Sementation

## Install Conda Env:
```
$ conda create -n cv python=3 accelerate imageio tqdm cudatoolkit=7.5 -y
``` 

## Test Cuda:
```
  python3 smcImgCuda.py input output steps hc hd m sdc sdd grayscale cardNumber

$ conda activate cv
$ python3 smcImgCuda.py images/school.png output/school-10-8-7-40-3-3.png 10 8 7 40 3 3 false 0
$ conda deactivate
```

## Description:
```
Check Assignment1.pdf
```
