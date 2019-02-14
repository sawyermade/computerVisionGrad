import imageio, os, sys
import numpy as np 

DEBUG = True

def parseConfig(cPath):
	if DEBUG: print('Parse Config Started...')
	coords = []
	with open(cPath) as cp:
		iPath = cp.readline().replace('\n', '')
		for line in cp:
			if len(line.split()) == 2:
				x, y = line.split()
				x, y = int(x), int(y)
				coords.append((x, y))
		
		# Opens image and returns img and coords
		img = imageio.imread(iPath)
		if DEBUG: print('Complete.')
		return img, coords

	return None, None

def insidePolygon(img, coords):
	if DEBUG: print('Inside Polygon Started...')
	rows, cols = img.shape[:2]
	
	pointList = []
	for x in range(cols):
		for y in range(rows):
			x1, y1 = coords[0]

			insideCount = 0
			for i in range(len(coords)+1):
				x2, y2 = coords[i % len(coords)]

				if y > min(y1,y2) and y <= max(y1,y2) and x <= max(x1,x2) and y1 != y2:
					inter = (y-y1)*(x2-x1)/(y2-y1)+x1

					if x1 == x2 or x <= inter:
						insideCount += 1
						

				x1, y1 = x2, y2

			if insideCount % 2 != 0:
				pointList.append((x,y))

	if DEBUG: print('Complete.')
	return pointList

def convertHv(Hv):
	V = np.array([
		[1, 0, 0],
		[0, 1, 0],
		[0, 0, 1]
	])

	H = np.append(Hv, 0)
	H = H.reshape((3,3))

	return V + H

def calcHomography(fromCoords, toCoords, damp):
	if DEBUG: print('Calc Homography Started...')
	# H = np.zeros((3,3), dtype=float)
	# H[0, 0] = 0
	# H[1, 1] = 0
	# H[2, 2] = 1
	# H = np.full((3, 3), 1, dtype=float)
	Hv = np.zeros((8), dtype=float)

	numIters = 0
	flag = True
	predictedPts = []
	while(flag):
		# print(i, H)
		Asum, Bsum, ADsum = np.zeros((8,8)), np.zeros((8,1)), np.zeros((8,8))
		for pf, pt in zip(fromCoords, toCoords):

			x, y = pt
			xi, yi = pf 

			H = convertHv(Hv)

			v = np.array([
				[x],
				[y],
				[1]
			])
			xv = np.matmul(H, v)
			d = 1.0/xv[2,0]
			xg, yg = xv[0,0]*d, xv[1,0]*d 
			# xg, yg = xv[0,0], xv[1,0]

			J = np.array([
				[x, y, 1, 0, 0, 0, -1.0*xg*x, -1.0*xg*y],
				[0, 0, 0, x, y, 1, -1.0*yg*x, -1.0*yg*y]
			]) * d

			# J = np.array([
			# 	[x, y, 1, 0, 0, 0, -1.0*xi*x, -1.0*xi*y],
			# 	[0, 0, 0, x, y, 1, -1.0*yi*x, -1.0*yi*y]
			# ]) * d
			ri = np.array([
				[xi-xg],
				[yi-yg]
			])
			# ri *= D
			# # J *= 1.0/d

			# # J = np.array([
			# # 	[x, y, 1, 0, 0, 0, -1.0*x*x, -1.0*x*y],
			# # 	[0, 0, 0, x, y, 1, -1.0*y*x, -1.0*y*y]
			# # ])
			# # ri = np.array([
			# # 	[xi-x],
			# # 	[yi-y]
			# # ])

			
			# D = 1.0/xv[2,0]	
			A = np.matmul(J.T, J)
			# AD = np.diagonal(A)
			# print(A)
			Asum += A
			# ADsum += AD

			
			B = np.matmul(J.T, ri)
			Bsum += B

			# DH = np.linalg.lstsq(A, B, rcond=None)[0]
			# DH = np.append(DH, 1)
			# H += DH.reshape((3,3))

		# Updates
		# AD = np.diagonal(Asum) * 1.0/np.sum(H[2])
		Ad = np.diagonal(Asum)
		Hd = np.linalg.lstsq(Asum + damp*Ad, Bsum, rcond=None)[0].reshape(8)
		# DH = np.append(DH, 1)
		# print(Hd.shape)
		Hv += Hd

		numIters += 1
		# if numIters > 1000000:
		# 	flag = False

		H = convertHv(Hv)
		newPts = testPoints(fromCoords, toCoords, H)
		if newPts == predictedPts:
			flag = False
		predictedPts = newPts
		

	if DEBUG: print('numIters =', numIters)
	if DEBUG: testOgPoints(fromCoords, toCoords, H)
	if DEBUG: print('Complete.')
	return H

def map2img(fromImg, toImg, toPoints, fromCoords, H):
	if DEBUG: print('Map2Img Started...')
	newImg = np.copy(toImg)
	minx, maxx = min(fromCoords, key=lambda x: x[0])[0], max(fromCoords, key=lambda x: x[0])[0]
	miny, maxy = min(fromCoords, key=lambda x: x[1])[1], max(fromCoords, key=lambda x: x[1])[1]
	# print(minx, maxx, miny, maxy)
	for p in toPoints:
		x, y = p
		v = np.array([
			[x],
			[y],
			[1]
		])

		xv = np.matmul(H, v)

		xi, yi = int(xv[0][0]/xv[2][0]), int(xv[1][0]/xv[2][0])
		if xi < minx: xi=minx
		if yi < miny: yi=miny
		if xi > maxx: xi=maxx
		if yi > maxy: yi=maxy

		try:
			newImg[y, x] = fromImg[yi, xi]
		except:
			# print(x, y, xi, yi)
			pass

	#
	# print(H[2,2])
	if DEBUG: print('Complete.')
	return newImg

def testOgPoints(fromCoords, toCoords, H):
	predictedPts = []
	for pf, pt in zip(fromCoords, toCoords):
		x, y = pt

		v = np.array([
			[x],
			[y],
			[1]
		])

		xv = np.matmul(H, v)

		xi, yi = xv[0][0]/xv[2][0], xv[1][0]/xv[2][0]
		xi, yi = int(xi), int(yi)
		predictedPts.append((xi, yi))
		if DEBUG: print(pf[0], pf[1], xi, yi)
	# if DEBUG: print()
	return predictedPts

def testPoints(fromCoords, toCoords, H):
	predictedPts = []
	for pf, pt in zip(fromCoords, toCoords):
		x, y = pt

		v = np.array([
			[x],
			[y],
			[1]
		])

		xv = np.matmul(H, v)

		xi, yi = xv[0][0]/xv[2][0], xv[1][0]/xv[2][0]
		xi, yi = round(xi, 4), round(yi, 4)
		predictedPts.append((xi, yi))

	return predictedPts


if __name__ == '__main__':
	# Gets info
	fromConfig = sys.argv[1]
	toConfig = sys.argv[2]
	if len(sys.argv) > 3:
		outPath = sys.argv[3]
	else:
		outPath = 'output/out.png'
	if len(sys.argv) > 4:
		damp = float(sys.argv[4])
	else:
		damp = 0.0

	# Parses coords
	fromImg, fromCoords = parseConfig(fromConfig)
	toImg, toCoords = parseConfig(toConfig)

	# Finds points within toImg polygon
	toPoints, fromPoints = insidePolygon(toImg, toCoords), insidePolygon(fromImg, fromCoords)
	
	# Gets homography matrix
	H = calcHomography(fromCoords, toCoords, damp)
	print('H = \n', H)

	# Map to image
	newImg = map2img(fromImg, toImg, toPoints, fromCoords, H)
	# newImg = map2img(fromImg, toImg, fromPoints, H)

	# Write new img
	imageio.imwrite(outPath, newImg)