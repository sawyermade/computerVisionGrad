import imageio, os, sys, math
import numpy as np

def firstOrderED(img, sigma):
	# Gets rows and cols
	rows, cols = img.shape

	# Creates placeholder
	dx = np.zeros((rows, cols), dtype=float)
	dy = np.zeros((rows, cols), dtype=float)
	dxdy = np.zeros((rows, cols), dtype=float)
	den = math.pow(sigma, 3) * math.sqrt(2.0 * math.pi)
	dist = sigma//2

	# Goes through every pixel horizontally, dx
	for i in range(rows):
		for j in range(cols):
			start, stop = j-dist, j+dist 
			if start < 0: start = 0
			if stop > cols-1: stop = cols-1

			total = 0
			for k in range(start, stop+1):
				x = k-j
				num = -1*x*math.exp(-1*x**2/2/sigma**2)
				total +=  img[i,k] * num / den

			dx[i,j] = total

	# Returns
	return dx, dy, dxdy

if __name__ == '__main__':
	# Args
	inFile = sys.argv[1]
	outFile = sys.argv[2]
	sigma = int(sys.argv[3])

	# Opens image
	img = imageio.imread(inFile, as_gray=True)

	# Runs 1st order edge detection
	dx, dy, dxdy = firstOrderED(img, sigma)

	# Saves
	imageio.imwrite('dx.png', dx)
	# imageio.imwrite('dy.png', dy)
	# imageio.imwrite('dxdy.png', dxdy)