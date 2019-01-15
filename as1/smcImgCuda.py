import numpy as np 
import imageio, os, math, sys
from tqdm import tqdm

# CUDA
from numba import cuda

# Converts color space
def rgb2xyz(im):
	# Creates empty matrix
	rows, cols, channels = im.shape
	xyzIm = np.zeros((rows, cols, 3), dtype=float)

	# Converts all pixels to XYZ
	for i in tqdm(range(rows)):
		for j in range(cols):
			# Grayscale
			if channels < 3:
				rgb = np.array([im[i,j,0], im[i,j,0], im[i,j,0]]) / 255.0

			# RGBa
			elif channels > 3:
				rgb = im[i, j, 0:3] / 255.0

			# RGB
			else:
				rgb = im[i, j] / 255.0
			
			# Calculates XYZ
			r, g, b = rgb
			if r > 0.04045: r = ((r + 0.055) / 1.055)**2.4
			else: r = r / 12.92

			if g > 0.04045: g = ((g + 0.055) / 1.055)**2.4
			else: g = g / 12.92

			if b > 0.04045: b = ((b + 0.055) / 1.055)**2.4
			else: b = b / 12.92

			r, g, b = r*100, g*100, b*100

			x = r * 0.4124 + g * 0.3576 + b * 0.1805
			y = r * 0.2126 + g * 0.7152 + b * 0.0722
			z = r * 0.0193 + g * 0.1192 + b * 0.9505

			xyzIm[i, j] = np.array([x, y, z])

			# xyzIm[i, j] = np.reshape(np.matmul(M, rgb), (3))
			if DEBUG and i == 0 and j == 0:
				print('RGB =', im[i, j])
				print('XYZ =', xyzIm[i, j])
				print('rgb = ', rgb)
				print('r, g, b = {}, {}, {}'.format(r, g, b))

	# Set XYZ image
	return xyzIm

def xyz2rgb(xyzIm):
	# Creates empty matrix
	rows, cols, channels = xyzIm.shape
	rgb = np.zeros((rows, cols, 3), dtype=np.uint8)

	# Converts to standard RGB
	for i in range(rows):
		for j in range(cols):
			X, Y, Z = xyzIm[i,j]

			X = X/100.0
			Y = Y/100.0
			Z = Z/100.0

			r = X *  3.2406 + Y * -1.5372 + Z * -0.4986
			g = X * -0.9689 + Y *  1.8758 + Z *  0.0415
			b = X *  0.0557 + Y * -0.2040 + Z *  1.0570

			if r > 0.0031308: r = 1.055 * (r**( 1 / 2.4 ) ) - 0.055
			else: r = 12.92 * r 

			if g > 0.0031308: g = 1.055 * (g**( 1 / 2.4 ) ) - 0.055
			else: g = 12.92 * g 

			if b > 0.0031308: b = 1.055 * (b**( 1 / 2.4 ) ) - 0.055
			else: b = 12.92 * b 

			r = np.uint8(round(r*255))
			g = np.uint8(round(g*255))
			b = np.uint8(round(b*255))

			rgb[i,j] = np.array([r,g,b], dtype=np.uint8)

	# Sets rgb
	return rgb

def xyz2lab(xyzIm):
	# Xn, Yn, Zn
	Xn, Yn, Zn = 0.950456*100, 1.0*100, 1.088754*100
	if DEBUG : print('Xn = {}, Yn = {}, Zn = {}'.format(Xn, Yn, Zn))

	# Creates Lab matrix
	rows, cols, _ = xyzIm.shape
	labImF = np.zeros((rows, cols, 3), dtype=float)
	labImD = np.zeros((rows, cols, 3), dtype=np.uint8)

	# Converts XYZ to Lab
	for i in tqdm(range(rows)):
		for j in range(cols):
			X, Y, Z = xyzIm[i, j]

			X, Y, Z = X/Xn, Y/Yn, Z/Zn

			if X > 0.008856: X = X**(1.0/3)
			else: X = (7.787*X) + (16.0/116)

			if Y > 0.008856: Y = Y**(1.0/3)
			else: Y = (7.787*Y) + (16.0/116)

			if Z > 0.008856: Z = Z**(1.0/3)
			else: Z = (7.787*Z) + (16.0/116)

			L = (116*Y)-16
			a = 500*(X-Y)
			b = 200*(Y-Z)

			Ld = np.uint8(round(L*255/100))
			ad = np.uint8(round(a+128))
			bd = np.uint8(round(b+128))

			labImF[i,j] = np.array([L, a, b])
			labImD[i,j] = [Ld, ad, bd]

			if DEBUG and i == 0 and j == 0: print(labImF[i,j], labImD[i,j])

	# Returns Lab
	return labImF, labImD

def lab2xyz(labIm):
	# Xn, Yn, Zn
	Xn, Yn, Zn = 0.950456*100, 1.0*100, 1.088754*100
	if DEBUG : print('Xn = {}, Yn = {}, Zn = {}'.format(Xn, Yn, Zn))

	# Creates Lab matrix
	rows, cols, _ = labIm.shape
	xyzIm = np.zeros((rows, cols, 3), dtype=float)

	# Converts lab 2 xyz
	for i in range(rows):
		for j in range(cols):
			L, a, b = labIm[i, j]

			Y = (L+16)/116.0
			X = a/500 + Y
			Z = Y - b/200

			if Y**3 > 0.008856: Y = Y**3
			else: Y = (Y - 16.0/116.0) / 7.787

			if X**3 > 0.008856: X = X**3
			else: X = (X - 16.0/116.0) / 7.787

			if Z**3 > 0.008856: Z = Z**3
			else: Z = (Z - 16.0/116.0) / 7.787

			X, Y, Z = X*Xn, Y*Yn, Z*Zn

			xyzIm[i,j] = np.array([X, Y, Z])

			if DEBUG and i == 0 and j == 0: print(xyzIm[i,j], xyzIm[i,j])

	# Sets xyz
	return xyzIm

def rgb2lab(rgb):
	return xyz2lab(rgb2xyz(rgb))

def lab2rgb(lab):
	return xyz2rgb(lab2xyz(lab))

@cuda.jit
def msCudaNaiveGS(img, newImg):
	# Gets coordinates i, j
	tx = cuda.threadIdx.x
	ty = cuda.threadIdx.y
	bx = cuda.blockIdx.x
	by = cuda.blockIdx.y
	bw = cuda.blockDim.x
	bh = cuda.blockDim.y
	i = tx + bx*bw
	j = ty + by*bh

	# If i, h inbounds
	if i < img.shape[0] and j < img.shape[1]:
		# Sets up vars needed
		meanSum, meanTotal, count = 0.0, 0.0, 0
		hc, hd, m, sdc, sdd = 8, 7, 20, 2, 2
		hci, hdi = 1.0/hc**2, 1.0/hd**2
		
		# Sets current pixel and compares against rest
		x = img[i,j,0]
		for k in range(img.shape[0]):
			for l in range(img.shape[1]):
				# Gets next pixel intensity
				xi = img[k,l,0]
				
				# Calculates spatial and intensity distances
				magHc = abs(x-xi)
				magHd = math.sqrt((i-k)**2 + (j-l)**2)

				# If within distances
				if magHc <= sdc*hc and magHd <= sdd*hd:
					count += 1

					# Calculates
					xxia, xxib, xxic = (x-xi)**2, (i-k)**2, (j-l)**2
					xxi = xxia*hci + xxib*hdi + xxic*hdi
					exp = math.exp(-0.5 * xxi)
					# magxi = math.sqrt(math.pow(xi, 2) + math.pow(k, 2) + math.pow(l, 2))

					# Adds to sums
					meanSum += xi * exp
					meanTotal += exp
		
		# Clustering
		if m < count:
			newImg[i,j,0] = meanSum / meanTotal
	
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
		tempImg = np.zeros((rows, cols, 1), np.uint8)
		for i in range(rows):
			for j in range(cols):
				tempImg[i,j,0] = img[i,j]
		# print(img.shape)
		img = tempImg
	

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
	if outDir:
		outDir = os.path.join(*outDir)
		if not os.path.exists(outDir):
			os.makedirs(outDir)
	# newImg = newImg.astype(np.uint8)
	imageio.imwrite(outPath, newImg)