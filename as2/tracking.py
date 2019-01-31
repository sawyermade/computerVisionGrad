import imageio, os, sys, math
import numpy as np

def autocorrelate(dx, dy, sigma, K):
	# Gets rows and cols
	rows, cols = dx.shape

	# Sigma shit
	dist = 2 * sigma

	# Creates eigen array
	eigen = np.zeros((rows, cols), dtype=float)
	g = 1.0/2.0/math.pi/sigma**2

	# Goes through all pixels
	for i in range(rows):
		for j in range(cols):
			# Sets boundaries
			startx, starty = i-dist, j-dist
			stopx, stopy = i+dist, j+dist

			if startx < 0: startx = 0
			if starty < 0: starty = 0
			if stopx > rows-1: stopx = rows-1
			if stopy > cols-1: stopy = cols-1

			# Goes through window
			dxdxsum, dydysum, dxdysum = 0, 0, 0
			for k in range(startx, stopx+1):
				for l in range(starty, stopy+1):
					x, y = k-i, l-j
					G = g*math.exp(-1.0*(x**2 + y**2)/2/sigma**2)
					dxdxsum += G*dx[k,l]**2
					dydysum += G*dy[k,l]**2
					dxdysum += G*dx[k,l]*dy[k,l]

			# Gets and sets min eigen vals
			summatrix = np.array([[dxdxsum, dxdysum], [dxdysum, dydysum]])
			eigen[i,j] = np.min(np.linalg.eigvals(summatrix))

	# Finds local max
	eigenMax = np.copy(eigen)
	for i in range(rows):
		for j in range(cols):
			# Sets boundaries
			startx, starty = i-dist, j-dist
			stopx, stopy = i+dist, j+dist

			if startx < 0: startx = 0
			if starty < 0: starty = 0
			if stopx > rows-1: stopx = rows-1
			if stopy > cols-1: stopy = cols-1

			# Goes through window
			flag = False
			for k in range(startx, stopx+1):
				for l in range(starty, stopy+1):
					if eigen[k,l] > eigen[i,j]:
						eigenMax[i,j] = 0
						flag = True
						break
				if flag: break
	# DEBUG
	print(np.count_nonzero(eigenMax))

	# Get top K eigens
	points = np.zeros((rows, cols), dtype=np.uint8)
	xl, yl = np.unravel_index(eigenMax.flatten().argsort()[-K:], eigenMax.shape)

	# Adds top K points
	for x, y in zip(xl, yl):
		points[x,y] = 255

	# Return eigen matrix
	return points

def firstOrderED(img, sigma):
	# Gets rows and cols
	rows, cols = img.shape

	# Sigma shit
	dist = 2 * sigma

	# Creates placeholder
	dx = np.zeros((rows, cols), dtype=float)
	dy = np.zeros((rows, cols), dtype=float)
	den = math.pow(sigma, 3) * math.sqrt(2.0 * math.pi)

	# Goes through every pixel horizontally, dx
	for i in range(rows):
		for j in range(cols):
			start, stop = j-dist, j+dist 
			if start < 0: start = 0
			if stop > cols-1: stop = cols-1

			total = 0
			for k in range(start, stop+1):
				x = k-j
				num = -1.0*x*math.exp(-1*x**2/2/sigma**2)
				total +=  img[i,k] * num / den

			# Sets pixel
			dx[i,j] = total

	# Goes through every pixel horizontally, dx
	for i in range(rows):
		for j in range(cols):
			start, stop = i-dist, i+dist
			if start < 0: start = 0
			if stop > rows-1: stop = rows-1

			total = 0
			for k in range(start, stop+1):
				x = k-i
				num = -1.0*x*math.exp(-1*x**2/2/sigma**2)
				total +=  img[k,j] * num / den

			# Sets pixel
			dy[i,j] = total

	# Returns
	return dx, dy

if __name__ == '__main__':
	# Args
	inFile = sys.argv[1]
	# outFile = sys.argv[2]
	sigma = int(sys.argv[2])

	# Opens image
	img = imageio.imread(inFile, as_gray=True)

	# Runs 1st order edge detection
	dx, dy = firstOrderED(img, sigma)

	# Saves
	imageio.imwrite('dx.png', dx)
	imageio.imwrite('dy.png', dy)

	# dx2 = imageio.imread('dx.png')
	# for i in range(dx.shape[0]):
	# 	for j in range(dx.shape[1]):
	# 		print(dx[i,j], dx2[i,j])

	# Gets eigen vectors
	K = 13
	eigen = autocorrelate(dx, dy, sigma, K)
	imageio.imwrite('eigen0.png', eigen)

	img = imageio.imread('moon_frames/frame1.png', as_gray=True)
	dx, dy = firstOrderED(img, sigma)
	eigen = autocorrelate(dx, dy, sigma, K)
	imageio.imwrite('eigen1.png', eigen)