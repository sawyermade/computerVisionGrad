import imageio, os, sys, math
import numpy as np
from tqdm import tqdm

def kalmanFilter(inPath, sigma, K):
	# Distance
	dist = 3*sigma

	# Gets file paths
	for root, dirs, files in os.walk(inPath):
		if files:
			fileList = [f for f in files if f.endswith('.png')]
			fileList.sort(key=lambda x: int(x.split('.')[0]))
			fileList = [os.path.join(root, f) for f in fileList]

	# Gets first frame and gets points
	prevFrame = imageio.imread(fileList[0], as_gray=True)
	prevFrame = np.pad(prevFrame, [(dist, dist),(dist, dist)], 'constant')
	dx, dy = firstOrderED(firstFrame, sigma)
	prevPoints = autocorrelateEigen(dx, dy, sigma, K)

	# Models
	A = np.array(
		[1, 0, 1, 0],
		[0, 1, 0, 1],
		[0, 0, 1, 0],
		[0, 0, 0, 1]
	)
	H = np.array(
		[1, 0, 0, 0],
		[0, 1, 0, 0]
	)
	R = np.array(
		[1, 0],
		[0, 1]
	)
	P0 = np.array(
		[9, 0, 0, 0],
		[0, 9, 0, 0],
		[0, 0, 25, 0],
		[0, 0, 0, 25]
	)
	Q = np.array(
		[0.25, 0, 0, 0],
		[0, 0.25, 0, 0],
		[0, 0, 0.25, 0],
		[0, 0, 0, 0.25]
	)

	# Goes through rest of images
	u, v = 0, 0
	for imgPath in fileList[1:]:
		# Gets image
		currFrame = imageio.imread(imgPath, as_gray=True)
		currFrame = np.pad(currFrame, [(dist, dist),(dist, dist)], 'constant')

		# Goes through prevPoints
		for pp in prevPoints:
			# Calcs intermediates
			St0 = np.array([pp[0], pp[1], u, v])
			St1_ = np.matmul(A, st0)
			P1_ = np.matmul(np.matmul(A, P0), A.T) + Q
			rows, cols = int(math.ceil(math.sqrt(P1_[0,0]))), int(math.ceil(math.sqrt(P1_[1,1])))

			# 


def autocorrelateEigen(dx, dy, sigma, K):	
	# Sigma shit
	dist = 3 * sigma

	# Creates eigen array
	eigen = np.zeros(dx.shape, dtype=float)

	# Creates kernel
	kern = np.zeros((2*dist+1, 2*dist+1), dtype=float)
	g = 1.0/2.0/math.pi/sigma**2
	for i in range(2*dist+1):
		for j in range(2*dist+1):
			x, y = i-dist, j-dist
			kern[i,j] = g*math.exp(-1.0*(x**2 + y**2)/2/sigma**2)

	# Goes through all pixels calcs eigens
	padx, pady = np.pad(dx, [(dist, dist), (dist, dist)], 'constant'), np.pad(dy, [(dist, dist), (dist, dist)], 'constant')
	rows, cols = padx.shape
	for i in tqdm(range(dist, rows-dist)):
		for j in range(dist, cols-dist):
			# Calcs window indices
			startx, starty = i-dist, j-dist
			stopx, stopy = i+dist, j+dist

			# Calcs window
			dxdxsum = np.sum(np.multiply(np.square(padx[startx:stopx+1, starty:stopy+1]), kern))
			dydysum = np.sum(np.multiply(np.square(pady[startx:stopx+1, starty:stopy+1]), kern))
			dxdysum = np.sum(np.multiply(np.multiply(padx[startx:stopx+1, starty:stopy+1], pady[startx:stopx+1, starty:stopy+1]), kern))

			# Calcs eigens
			summatrix = np.array([[dxdxsum, dxdysum], [dxdysum, dydysum]])
			eigen[i-dist,j-dist] = np.min(np.linalg.eigvals(summatrix))

	# Finds local max
	eigenMax = np.copy(eigen)
	rows, cols = eigen.shape
	for i in range(rows):
		for j in range(cols):
			# Sets boundaries
			startx, starty = i-dist, j-dist
			stopx, stopy = i+dist, j+dist

			if startx < 0: startx = 0
			if starty < 0: starty = 0
			if stopx > rows-1: stopx = rows-1
			if stopy > cols-1: stopy = cols-1

			# Finds max val in window, checks if current is max
			maxVal = np.max(eigen[startx:stopx+1, starty:stopy+1])
			if maxVal > eigen[i,j]: eigenMax[i,j] = 0

	# Get top K eigens
	points = np.zeros(eigenMax.shape, dtype=np.uint8)
	xl, yl = np.unravel_index(eigenMax.flatten().argsort()[-K:], eigenMax.shape)

	# Adds top K points
	pointList = []
	for x, y in zip(xl, yl):
		points[x,y] = 255
		pointList.append((x, y))

	# Return eigen matrix
	return pointList

def firstOrderED(img, sigma):
	# Sigma shit
	dist = 3 * sigma

	# Creates kernel
	kern = np.zeros((2*dist+1), dtype=float)
	den = math.pow(sigma, 3) * math.sqrt(2.0 * math.pi)
	for i in range(2*dist+1):
		x = i - dist
		kern[i] = -1.0*x*math.exp(-1*x**2/2/sigma**2) / den

	# Goes through every pixel horizontally, dx
	pad = np.pad(img, [(dist, dist), (dist, dist)], 'constant')
	rows, cols = pad.shape
	dx = np.zeros((img.shape[0], img.shape[1]), dtype=float)
	for i in range(dist, rows-dist):
		for j in range(dist, cols-dist):
			# Calcs window indices
			start, stop = j-dist, j+dist

			# Sets pixel
			total = np.multiply(pad[i, start:stop+1], kern)
			dx[i-dist,j-dist] = np.sum(total)

	# Goes through every pixel vertically, dy
	padt = pad.T
	rows, cols = padt.shape
	dy = np.zeros((img.shape[1], img.shape[0]), dtype=float)
	for i in range(dist, rows-dist):
		for j in range(dist, cols-dist):
			# Calcs window indices
			start, stop = j-dist, j+dist

			# Sets pixel
			total = np.multiply(padt[i, start:stop+1], kern)
			dy[i-dist,j-dist] = np.sum(total)

	# Returns
	return dx, dy.T

if __name__ == '__main__':
	# Args
	inFile = sys.argv[1]
	sigma = int(sys.argv[2])
	K = int(sys.argv[3])

	# Opens image
	img = imageio.imread(inFile, as_gray=True)

	# Runs 1st order edge detection
	dx, dy = firstOrderED(img, sigma)

	# # Saves
	# imageio.imwrite('dx.png', dx)
	# imageio.imwrite('dy.png', dy)

	# Gets eigen vectors
	eigen = autocorrelateEigen(dx, dy, sigma, K)
	imageio.imwrite('eigen0.png', eigen)

	# img = imageio.imread('moon_frames/frame1.png', as_gray=True)
	# dx, dy = firstOrderED(img, sigma)
	# eigen = autocorrelate(dx, dy, sigma, K)
	# imageio.imwrite('eigen1.png', eigen)