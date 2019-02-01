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
	dx, dy = firstOrderED(prevFrame, sigma)
	prevPoints = autocorrelateEigen(dx, dy, sigma, K)
	prevPoints = [(p[0], p[1], 0, 0) for p in prevPoints]


	# Models
	A = np.array([
		[1, 0, 1, 0],
		[0, 1, 0, 1],
		[0, 0, 1, 0],
		[0, 0, 0, 1]
	])
	H = np.array([
		[1, 0, 0, 0],
		[0, 1, 0, 0]
	])
	R = np.array([
		[1, 0],
		[0, 1]
	])
	P00 = np.array([
		[9, 0, 0, 0],
		[0, 9, 0, 0],
		[0, 0, 25, 0],
		[0, 0, 0, 25]
	])
	Q = np.array([
		[0.25, 0, 0, 0],
		[0, 0.25, 0, 0],
		[0, 0, 0.25, 0],
		[0, 0, 0, 0.25]
	])

	# Make initial P0 array
	P0List = np.ndarray((K, 4, 4))
	for i in range(K):
		P0List[i] = np.copy(P00)
	# print(P0List)

	# Goes through rest of images
	for imgPath in tqdm(fileList[1:10]):
		# Gets image
		currFrame = imageio.imread(imgPath, as_gray=True)

		# Goes through prevPoints
		newPoints = []
		newP0List = np.ndarray((len(prevPoints), 4, 4))
		count = 0
		for pp, P0 in zip(prevPoints, P0List):
			# Calcs intermediates
			St0 = np.array([pp[0], pp[1], pp[2], pp[3]])
			St1_ = np.matmul(A, St0)
			P1_ = np.matmul(np.matmul(A, P0), A.T) + Q
			rows, cols = int(math.sqrt(math.ceil(P1_[0,0]))), int(math.sqrt(math.ceil(P1_[1,1])))
			sigx, sigy = 3*rows, 3*cols

			# Autocorrelates point with prev image and current image
			Mt1 = autocorrelate(prevFrame, currFrame, St1_, sigx, sigy, sigma)

			#DEBUG
			# print('old = ({}, {})  new = ({}, {})'.format(St0[0], St0[1], Mt1[0], Mt1[1]))

			# Calculate St1, P1
			K = np.matmul(np.matmul(P1_, H.T), np.linalg.inv(np.matmul(np.matmul(H, P1_), H.T) + R))
			P1 = P1_ + np.matmul(np.matmul(K, H), P1_)
			St1 = St1_ + np.matmul(K, Mt1 - np.matmul(H, St1_))
			# St0[2], St0[3] = St1[2], St1[3]
			# St1 = np.matmul(A, St0)
			# print(St1_, St1)

			# Adds to newPoints and P0List
			newPoints.append((int(St1[0]), int(St1[1]), int(St1[2]), int(St1[3])))
			newP0List[count] = P1
			count += 1

		prevPoints = newPoints
		P0List = newP0List
		prevFrame = currFrame
		# print()

def autocorrelate(img1, img2, St1_, rows, cols, sigma):	
	# Sigma shit
	dist = 3 * sigma

	# Creates kernel
	kern = np.zeros((2*dist+1, 2*dist+1), dtype=float)
	g = 1.0/2.0/math.pi/sigma**2
	for i in range(2*dist+1):
		for j in range(2*dist+1):
			x, y = i-dist, j-dist
			kern[i,j] = g*math.exp(-1.0*(x**2 + y**2)/2/sigma**2)

	# Calcs windows coords
	winxs = St1_[0] - rows//2
	winys = St1_[1] - cols//2
	winxe = winxs + rows
	winye = winys + cols
	if winxs < 0: winxs = 0
	if winys < 0: winys = 0
	if winxe < 0: winxe = 0
	if winye < 0: winye = 0
	if winxs > img1.shape[0]: winxs = img1.shape[0]
	if winys > img1.shape[1]: winys = img1.shape[1]
	if winxe > img1.shape[0]: winxe = img1.shape[0]
	if winye > img1.shape[1]: winye = img1.shape[1]

	# Goes through all pixels calcs eigens
	sumMat = np.full(img1.shape, sys.maxsize, dtype=float)
	# coordMat = np.zeros((winxe-winxs, winye-winys, 2), dtype=int)
	pad1, pad2 = np.pad(img1, [(dist, dist), (dist, dist)], 'constant'), np.pad(img2, [(dist, dist), (dist, dist)], 'constant')
	cx = 0
	for i in range(winxs+dist, winxe-dist):
		cy = 0
		for j in range(winys+dist, winye-dist):
			# Calcs window indices
			startx, starty = i-dist, j-dist
			stopx, stopy = i+dist, j+dist

			# Gets difference
			diffMat = np.subtract(pad1[startx:stopx+1, starty:stopy+1], pad2[startx:stopx+1, starty:stopy+1])

			# Get sum with gauss kernel
			sumMat[i-dist, j-dist] = np.sum(np.multiply(np.square(diffMat), kern))

	# Finds min coords
	newx, newy = np.unravel_index(np.argmin(sumMat), sumMat.shape)

	# Returns new measurement
	print(newx, newy)
	return np.array([newx, newy]).reshape((2,))

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
	for i in range(dist, rows-dist):
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
	inPath = sys.argv[1]
	sigma = int(sys.argv[2])
	K = int(sys.argv[3])

	# # Opens image
	# img = imageio.imread(inFile, as_gray=True)

	# # Runs 1st order edge detection
	# dx, dy = firstOrderED(img, sigma)

	# # Saves
	# imageio.imwrite('dx.png', dx)
	# imageio.imwrite('dy.png', dy)

	# Gets eigen vectors
	# eigen = autocorrelateEigen(dx, dy, sigma, K)
	# imageio.imwrite('eigen0.png', eigen)

	# img = imageio.imread('moon_frames/frame1.png', as_gray=True)
	# dx, dy = firstOrderED(img, sigma)
	# eigen = autocorrelate(dx, dy, sigma, K)
	# imageio.imwrite('eigen1.png', eigen)

	kalmanFilter(inPath, sigma, K)