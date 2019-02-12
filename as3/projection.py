import imageio, os, sys
import numpy as np 

if __name__ == '__main__':
	# Gets info
	toImgPath = sys.argv[1]
	toCoords = sys.argv[2]
	fromImgPath = sys.argv[3]
	toImg = imageio.imread(toImgPath)
	fromImg = imageio.imread(fromImgPath)

	# Checks if from image has coords or not
	if len(sys.argv) > 4:
		fromCoords = sys.argv[4]
	else:
		fromCoords = '0,0,{},{}'.format(fromImg.shape[1], fromImg.shape[0])

	# Parses coords
	toCoords = toCoords.replace(' ', '').split(',')
	fromCoords = fromCoords.replace(' ', '').split(',')
	toCoords = [int(i) for i in toCoords]
	fromCoords = [int(i) for i in fromCoords]

	# Finds points within toImg polygon
	toPoints = insidePolygon(toImg, toCoords)
	fromPoints = insidePolygon(fromImg, fromCoords)