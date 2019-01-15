import numpy as np 
import imageio, os, math, sys
from tqdm import tqdm

# CUDA
from numba import cuda 

@cuda.jit
def msCudaNaiveGS(img, newImg):
	# Gets coords and checks if inbounds
	tx = cuda.threadIdx.x
	ty = cuda.threadIdx.y
	bx = cuda.blockIdx.x
	by = cuda.blockIdx.y
	bw = cuda.blockDim.x
	bh = cuda.blockDim.y
	i = tx + bx*bw
	j = ty + by*bh

	# i, j = cuda.grid(2)
	if i < img.shape[0] and j < img.shape[1]:
		meanSum, meanTotal = 0.0, 0.0
		count = 0
		hc, hd, m, sdc, sdd = 8, 7, 20, 2, 2

		# if i == 228: print(j)
		x = img[i,j,0]

		for k in range(img.shape[0]):
			for l in range(img.shape[1]):
				xi = img[k,l,0]

				magHc = math.sqrt(math.pow(x-xi, 2))
				magHd = math.sqrt(math.pow(i-k, 2) + math.pow(j-l, 2))

				if magHc <= sdc*hc and magHd <= sdd*hd:
					count += 1

					# xxi = (x-xi)**2 + (i-k)**2 + (j-l)**2
					xxi = math.pow(x-xi, 2) + math.pow(i-k, 2) + math.pow(j-l, 2)
					exp1 = math.exp(-0.5 * xxi / hc**2) + math.exp(-0.5 * xxi / hd**2)
					# exp2 = math.exp(-0.5 * xxi / (hc**2 + hd**2))
					# exp3 = math.exp(-0.5 * xxi / (hc**2 * hd**2))
					magxi = math.sqrt(math.pow(xi, 2) + math.pow(k, 2) + math.pow(l, 2))
					
					# if i == 0 and j % 50 == 0:
					# 	print(i, j, i*exp1, j*exp1)

					meanSum += xi * exp1
					meanTotal += exp1
		
		# if i == 228 and j == 228:
		# 	print(meanSum, meanTotal, meanSum / meanTotal, meanSum / meanTotal - x, x)
		if m < count:
			# print(count)
			newImg[i,j,0] = int(meanSum / meanTotal)
	
# Main
if __name__ == '__main__':
	# Gets args
	_, inPath, outPath, cardNumber, steps = sys.argv

	# Sets cuda device
	os.environ['CUDA_VISIBLE_DEVICES'] = cardNumber

	# Opens image
	img = imageio.imread(inPath)
	rows, cols = img.shape[0], img.shape[1]
	# print('img shape = {}'.format(img.shape))

	# Checks for single channel grayscale
	if len(img.shape) < 3:
		for i in range(rows):
			for j in range(cols):
				img[i,j] = np.array([img[i,j]])
		# print(img.shape)

	

	# Runs steps
	grayscale = True
	if grayscale and img.shape[2] > 1:
		tempImg = np.zeros((rows, cols, 1), np.uint8)
		for i in range(rows):
			for j in range(cols):
				tempImg[i,j,0] = img[i,j,0]
				# print(tempImg[i,j,0])
		
		# Sets grayscale image
		img = tempImg
		# print(img.shape)

	else:
		tempImg = np.zeros((rows, cols, 3), np.uint8)
		for i in range(rows):
			for j in range(cols):
				tempImg[i,j] = img[i,j]
				# print(tempImg[i,j])
		
		# Sets color image
		img = tempImg
		# print(img.shape)

	# Copies img
	newImg = np.copy(img)
	
	# Goes through steps
	TPB = 16
	hc, hd, m = 7, 8, 40
	for step in tqdm(range(int(steps))):
	# for step in range(int(steps)):
		# Copies to card
		imgCuda = cuda.to_device(newImg)
		newImgCuda = cuda.to_device(newImg)

		# Config blocks
		tpb = (TPB, TPB)
		bgx = int(math.ceil(rows / TPB))
		bgy = int(math.ceil(cols / TPB))
		bgrid = (bgx, bgy)

		# Run kernel
		if grayscale:
			msCudaNaiveGS[bgrid, tpb](imgCuda, newImgCuda)

		# Copy back
		newImg = newImgCuda.copy_to_host()

	# # Check vars
	# skip = 1000
	# for i in range(rows):
	# 	for j in range(cols):
	# 		if i % skip == 0 and i % skip == 0:
	# 			print(img[i,j,0], newImg[i,j,0])

	# Saves img
	outDir = outPath.split(os.sep)[:-1]
	outDir = os.path.join(*outDir)
	if not os.path.exists(outDir):
		os.makedirs(outDir)
	# newImg = newImg.astype(np.uint8)
	imageio.imwrite(outPath, newImg)