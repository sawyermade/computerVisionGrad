import os, sys, cv2 as cv, numpy as np, math

ONEMM = 3.7795275591

def swapColor(color):
	if color == 255: return 0
	else: return 255

def main():
	# Gets args
	outPath = sys.argv[1]
	iw, ih = int(sys.argv[2]), int(sys.argv[3])
	boxSize = int(sys.argv[4])

	# Gets pixel width/height
	ipw, iph = math.ceil(iw*ONEMM), math.ceil(ih*ONEMM)
	boxPSize = math.ceil(boxSize*ONEMM)
	numBoxes = math.ceil(ipw / boxPSize)
	# print(iw, ih, boxSize, iw/boxSize)

	# Creates img
	img = np.zeros((ipw, iph), dtype=np.uint8)

	# Creates checker board
	colorRow, colorCol = 255, 255 
	for i in range(iph):
		if i % boxPSize == 0:
			colorRow = swapColor(colorRow)

		colorCol = swapColor(colorRow)
		for j in range(ipw):
			if j % boxPSize == 0:
				colorCol = swapColor(colorCol)

			img[i, j] = colorCol

	# Writes img
	cv.imwrite(outPath, img)

if __name__ == '__main__':
	main()