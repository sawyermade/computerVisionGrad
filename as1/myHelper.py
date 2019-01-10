import numpy as np 
import imageio, os

DEBUG = False

class maFrignImg:
	# Init stuff
	def __init__(self, inPath=None, outPath=None, cSpace=None):
		self.inPath = inPath
		self.outPath = outPath
		self.cSpace = cSpace
		self.rgb = None
		self.lab = None
		self.labd = None
		self.xyz = None
		self.backup = None

		if self.inPath:
			self.rgb = imageio.imread(inPath)

			if self.cSpace:
				if self.cSpace == 'lab': self.lab, self.labd = self.rgb2lab(self.rgb)
				elif self.cSpace == 'xyz': self.xyz = self.rgb2xyz(self.rgb)

	def rgb2xyz(self):
		# Creates empty matrix
		im = self.rgb
		rows, cols, channels = im.shape
		xyzIm = np.zeros((rows, cols, 3), dtype=float)

		# Converts all pixels to XYZ
		for i in range(rows):
			for j in range(cols):
				# Grayscale
				if channels < 3:
					rgb = np.array([im[i,j,0], im[i,j,0], im[i,j,0]]) / 255

				# RGBa
				elif channels > 3:
					rgb = im[i, j, 0:3] / 255

				# RGB
				else:
					rgb = im[i, j] / 255
				
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
		self.xyz = xyzIm
		return True

	def xyz2lab(self):
		xyzIm = self.xyz
		# Xn, Yn, Zn
		Xn, Yn, Zn = 0.950456*100, 1.0*100, 1.088754*100
		if DEBUG : print('Xn = {}, Yn = {}, Zn = {}'.format(Xn, Yn, Zn))

		# Creates Lab matrix
		rows, cols, _ = xyzIm.shape
		labImF = np.zeros((rows, cols, 3), dtype=float)
		labImD = np.zeros((rows, cols, 3), dtype=np.uint8)

		# Converts XYZ to Lab
		for i in range(rows):
			for j in range(cols):
				X, Y, Z = xyzIm[i, j]

				X, Y, Z = X/Xn, Y/Yn, Z/Zn

				if X > 0.008856: X = X**(1/3)
				else: X = (7.787*X) + (16/116)

				if Y > 0.008856: Y = Y**(1/3)
				else: Y = (7.787*Y) + (16/116)

				if Z > 0.008856: Z = Z**(1/3)
				else: Z = (7.787*Z) + (16/116)

				L = (116*Y)-16
				a = 500*(X-Y)
				b = 200*(Y-Z)

				Ld = np.uint8(round(L*255/100))
				ad = np.uint8(round(a+128))
				bd = np.uint8(round(b+128))

				labImF[i,j] = np.array([L, a, b])
				labImD[i,j] = [Ld, ad, bd]

				if DEBUG and i == 1 and j == 1: print(labImF[i,j], labImD[i,j])

		# Returns Lab
		self.lab, self.labd = labImF, labImD
		return True

	def rgb2lab(self):
		self.rgb2xyz()
		self.xyz2lab()
		return True

	def open(self, path, cSpace=None):
		self.__init__(inPath=path, cSpace=cSpace)

	def save(self, path=None, im=None):
		if path:
			self.outPath = path

		if self.outPath:
			self.checkOutDir()
			if im is not None: imageio.imwrite(self.outPath, im)
			else: imageio.imwrite(self.outPath, self.rgb)
			return True
		print('No save path, file not saved.')
		return False

	def checkOutDir(self):
		outDir = self.outPath.split(os.sep)[:-1]
		outDir = os.path.join(*outDir)
		if not os.path.exists(outDir): os.makedirs(outDir)

def convertXYZ(im):
	# Creates empty matrix
	rows, cols, channels = im.shape
	xyzIm = np.zeros((rows, cols, 3), dtype=float)

	# Converts all pixels to XYZ
	for i in range(rows):
		for j in range(cols):
			# Grayscale
			if channels < 3:
				rgb = np.array([im[i,j,0], im[i,j,0], im[i,j,0]]) / 255

			# RGBa
			elif channels > 3:
				rgb = im[i, j, 0:3] / 255

			# RGB
			else:
				rgb = im[i, j] / 255
			
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

	# Return XYZ image
	return xyzIm

def convertLab(xyzIm):
	# Xn, Yn, Zn
	Xn, Yn, Zn = 0.950456*100, 1.0*100, 1.088754*100
	if DEBUG : print('Xn = {}, Yn = {}, Zn = {}'.format(Xn, Yn, Zn))

	# Creates Lab matrix
	rows, cols, channels = xyzIm.shape
	labImF = np.zeros((rows, cols, 3), dtype=float)
	labImD = np.zeros((rows, cols, 3), dtype=np.uint8)

	# Converts XYZ to Lab
	for i in range(rows):
		for j in range(cols):
			X, Y, Z = xyzIm[i, j]

			X, Y, Z = X/Xn, Y/Yn, Z/Zn

			if X > 0.008856: X = X**(1/3)
			else: X = (7.787*X) + (16/116)

			if Y > 0.008856: Y = Y**(1/3)
			else: Y = (7.787*Y) + (16/116)

			if Z > 0.008856: Z = Z**(1/3)
			else: Z = (7.787*Z) + (16/116)

			L = (116*Y)-16
			a = 500*(X-Y)
			b = 200*(Y-Z)

			Ld = np.uint8(round(L*255/100))
			ad = np.uint8(round(a+128))
			bd = np.uint8(round(b+128))

			labImF[i,j] = np.array([L, a, b])
			labImD[i,j] = [Ld, ad, bd]

			if DEBUG and i == 1 and j == 1: print(labImF[i,j], labImD[i,j])

	# Returns Lab
	return labImF, labImD

def rgb2lab(im):
	return convertLab(convertXYZ(im))

def imgRead(path):
	return imageio.imread(path)

def imWrite(path, im):
	checkOutDir(path)
	return imageio.imwrite(path, im)

def checkOutDir(path):
	outDir = path.split(os.sep)[:-1]
	outDir = os.path.join(*outDir)
	if not os.path.exists(outDir): os.makedirs(outDir)

def main():
	# Test 1
	test = imgRead('images/test-00.jpg')
	xyz = convertXYZ(test)
	labF, labD  = rgb2lab(test)
	imWrite('output/test/xyz.png', xyz)
	imWrite('output/test/lab.png', labD)

	# Test 2
	test = maFrignImg('images/test-00.jpg')
	test.rgb2lab()
	test.save('output/test/test-00-lab.png', test.labd)
	test.save('output/test/test-00.png')


if __name__ == '__main__':
	DEBUG = True
	main()