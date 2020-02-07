import imageio, os, sys, math
import numpy as np
from tqdm import tqdm

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
		points[x,y] = eigenMax[x,y]
		pointList.append((x, y))

	# Return eigen matrix
	return pointList, points

def firstOrderED(img, sigma):
	# Sigma stuff
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

def kalmanFilter(inPath, sigma, KP, outPath='output/'):
	# Creates output folder
	if not os.path.exists(outPath):
		os.makedirs(outPath)

	# Distance
	dist = 3*sigma

	# Gets file paths
	for root, dirs, files in os.walk(inPath):
		if files:
			fileList = [f for f in files if f.endswith('.png') or f.endswith('.jpg')]
			fileList.sort(key=lambda x: int(x.split('.')[0]))
			fileList = [os.path.join(root, f) for f in fileList]
			# print(fileList)

	# Gets first frame and gets points
	prevFrame = imageio.imread(fileList[0], as_gray=True)
	dx, dy = firstOrderED(prevFrame, sigma)
	prevPoints, points = autocorrelateEigen(dx, dy, sigma, KP)
	prevPoints = [(p[0], p[1], 0, 0) for p in prevPoints]
	# imageio.imwrite('eigen0.png', points)

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
	], dtype=float)
	Q = np.array([
		[0.25, 0, 0, 0],
		[0, 0.25, 0, 0],
		[0, 0, 0.25, 0],
		[0, 0, 0, 0.25]
	])
	I = np.array([
		[1, 0, 0, 0],
		[0, 1, 0, 0],
		[0, 0, 1, 0],
		[0, 0, 0, 1]
	])
	AP = np.array([
		[.65, 0, 1, 0],
		[0, .65, 0, 1],
		[0, 0, .65, 0],
		[0, 0, 0, .65]
	])

	# Make initial P0 array
	P0List = np.ndarray((KP, 4, 4), dtype=float)
	for i in range(KP):
		P0List[i] = P00

	# Goes through rest of images
	ic = 0
	updateFrames = 10
	for imgPath in tqdm(fileList[1:]):
		# Gets image
		currFrame = imageio.imread(imgPath, as_gray=True)

		#DEBUG
		# dx, dy = firstOrderED(currFrame, sigma)
		# pss, ps = autocorrelateEigen(dx, dy, sigma, KP)
		pss2 = []

		# Goes through prevPoints
		newPoints = []
		newP0List = np.ndarray((len(prevPoints), 4, 4))
		count = 0
		for pp, P0 in zip(prevPoints, P0List):
			# Calcs intermediates
			St0 = np.array([pp[0], pp[1], pp[2], pp[3]], dtype=float)
			St1_ = np.matmul(A, St0).astype(float)
			P1_ = np.matmul(np.matmul(A, P0), A.T) + Q
			rows, cols = int(math.ceil(math.sqrt(P1_[0,0]))), int(math.ceil(math.sqrt(P1_[1,1])))
			ui, vi = int(math.ceil(math.sqrt(P1_[2,2]))), int(math.ceil(math.sqrt(P1_[3,3])))
			sigx, sigy = 3*rows, 3*cols

			# Autocorrelates point with prev image and current image
			Mt1 = autocorrelate(prevFrame, currFrame, St0, St1_, sigx, sigy, ui, vi, sigma).astype(float)

			# Calculate St1, P1
			K = np.matmul(np.matmul(P1_, H.T), np.linalg.inv(np.matmul(np.matmul(H, P1_), H.T) + R))
			P1 = P1_ + np.matmul(np.matmul(K, H), P1_)
			# P1 = np.matmul((I - np.matmul(K, H)), P1_)
			St1 = St1_ + np.matmul(K, np.subtract(Mt1, np.matmul(H, St1_)))

			# Adds to newPoints and P0List
			newPoints.append((St1[0], St1[1], St1[2], St1[3]))
			pss2.append((int(St1[0]), int(St1[1])))
			newP0List[count] = P1
			count += 1

		# State updates
		P0List = newP0List
		if ic != 0 and ic % updateFrames == 0:
			prevPoints = newPoints
			prevFrame = np.copy(currFrame)
			updateFrames += 5
			# print(K)
			# prevFrame = currFrame
			# dx, dy = firstOrderED(prevFrame, sigma)
			# prevPoints, points = autocorrelateEigen(dx, dy, sigma, KP)
			# prevPoints = [(p[0], p[1], 0, 0) for p in prevPoints]


		# #DEBUG Writes points
		# io = np.zeros((currFrame.shape[0], currFrame.shape[1], 3), dtype=np.uint8)
		# for i in range(currFrame.shape[0]):
		# 	for j in range(currFrame.shape[1]):
		# 		io[i,j] = [currFrame[i,j]]*3
		# for p1, p2 in zip(pss, pss2):
		# 	# print(p1, p2)
		# 	io[p1[0], p1[1]] = [0, 255, 0]
		# 	io[p2[0], p2[1]] = [255, 0, 0]
		# imageio.imwrite('output/{}.png'.format(ic), io)

		# Write Points
		io = np.zeros((currFrame.shape[0], currFrame.shape[1], 3), dtype=np.uint8)
		for i in range(currFrame.shape[0]):
			for j in range(currFrame.shape[1]):
				io[i,j] = [currFrame[i,j]]*3
		for p1 in pss2:
			io[p1[0], p1[1]] = [255, 0, 0]

		imageio.imwrite(os.path.join(outPath, '{}.png'.format(ic)), io)
		ic += 1

def autocorrelate(img1, img2, St0, St1_, sigx, sigy, ui, vi, sigma):
	# Pads images
	dist = 3 * sigma 
	pad1 = np.pad(img1, [(dist, dist), (dist, dist)], 'constant')
	pad2 = np.pad(img2, [(dist, dist), (dist, dist)], 'constant')
	rows, cols = img1.shape

	# Creates kernel
	kern = np.zeros((2*dist+1, 2*dist+1), dtype=float)
	g = 1.0/2.0/math.pi/sigma**2
	for i in range(2*dist+1):
		for j in range(2*dist+1):
			x, y = i-dist, j-dist
			kern[i,j] = g*math.exp(-1.0*(x**2 + y**2)/2/sigma**2)
	# print(kern)

	# Sets up original patch
	# Top left corner bounds check
	px1, py1 = St0[:2]
	if px1 < 0: px1=0
	if py1 < 0: py1=0
	if px1 > rows-1: px1=rows-1
	if py1 > cols-1: py1=cols-1
	px1, py1 = px1+dist, py1+dist
	xa1, ya1 = int(px1-dist), int(py1-dist)
	# Bottom right corner bounds check
	xa2, ya2 = int(px1+dist)+1, int(py1+dist)+1
	# Patch1 slice
	patch1 = pad1[xa1:xa2, ya1:ya2]

	# Sets up covariance patch to search
	# Top left corner bounds check
	px2, py2 = St1_[:2]
	if px2 < 0: px2=0
	if py2 < 0: py2=0
	if px2 > rows-1: px2=rows-1
	if py2 > cols-1: py2=cols-1
	xb1, yb1 = int(px2-sigx//2), int(py2-sigy//2)
	if xb1 < 0: xb1=0
	if yb1 < 0: yb1=0
	if xb1 > rows-1: xb2=rows-1
	if yb1 > cols-1: yb2=cols-1
	# Bottom right corner bounds check
	xb2, yb2 = int(px2+sigx//2)+1, int(py2+sigy//2)+1
	if xb2 > rows: xb2=rows
	if yb2 > cols: yb2=cols
	if xb2 < 0: xb1=1
	if yb2 < 0: yb1=1
	# Adjusts to padding
	px2, py2 = px2+dist, py2+dist
	xb1, yb1 = xb1+dist, yb1+dist 
	xb2, yb2 = xb2+dist, yb2+dist

	#DEBUG
	# print(St0[:2], St1_[:2])
	# print(xa1, xa2, ya1, ya2)
	# print(xb1, xb2, yb1, yb2)
	# print()

	# Goes through covariance patch pixel by pixel
	sumMat = np.full(img1.shape, sys.maxsize//3, dtype=float)
	for i in range(xb1, xb2):
		for j in range(yb1, yb2):
			# Gets patch2 slice
			patch2 = pad2[i-dist:i+dist+1, j-dist:j+dist+1]

			# Calcs sum
			diffSquared = np.square(np.subtract(patch1, patch2))
			sumMat[i-dist, j-dist] = np.sum(np.multiply(diffSquared, kern))

	# Finds min coords
	newx, newy = np.unravel_index(np.argmin(sumMat), sumMat.shape)

	# Returns new measurement
	return np.array([newx, newy]).reshape((2,))

if __name__ == '__main__':
	# Args
	inPath = sys.argv[1]
	outPath = sys.argv[2]
	sigma = int(sys.argv[3])
	K = int(sys.argv[4])

	# # Opens image
	# img = imageio.imread(inPath, as_gray=True)

	# # Runs 1st order edge detection
	# dx, dy = firstOrderED(img, sigma)

	# # Saves
	# imageio.imwrite('dx.png', dx)
	# imageio.imwrite('dy.png', dy)

	# Gets eigen vectors
	# pts, eigen = autocorrelateEigen(dx, dy, sigma, K)
	# imageio.imwrite('eigen0.png', eigen)

	# img = imageio.imread('moon_frames/frame1.png', as_gray=True)
	# dx, dy = firstOrderED(img, sigma)
	# eigen = autocorrelate(dx, dy, sigma, K)
	# imageio.imwrite('eigen1.png', eigen)

	kalmanFilter(inPath, sigma, K, outPath)