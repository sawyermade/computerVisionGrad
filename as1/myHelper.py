import numpy as np 
import imageio, os, math, sys
from tqdm import tqdm
from multiprocessing import Pool

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
		self.meanshift = None

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
		self.xyz = xyzIm
		return True

	def xyz2rgb(self):
		# Creates empty matrix
		xyzIm = self.xyz
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
		self.rgb = rgb
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
		self.lab, self.labd = labImF, labImD
		return True

	def lab2xyz(self):
		labIm = self.lab
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
		self.xyz = xyzIm
		return True

	def rgb2lab(self):
		self.rgb2xyz()
		self.xyz2lab()
		return True

	def lab2rgb(self):
		self.lab2xyz()
		self.xyz2rgb()
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

	def hfx(self, x, xi, h):
		mag = [(i-j)**2 for i, j in zip(x, xi)]
		mag = math.sqrt(sum(mag))

		if mag <= 3*h: return True
		else: return False

	def threadMS(self, x, rows, cols, ogIm, newIm, hc, hd, grayScale):
		for k in range(rows):
			for l in range(cols):
				if grayScale:
					xi = [ogIm[k,l][0]]
				else:
					xi = list(ogIm[k,l])

				if self.hfx(x, xi, hc) and self.hfx([i,j], [k,l], hd):
					count += 1
					
					if grayScale:
						vec1, vec2 = x, xi
						vec1 += [i, j] 
						vec2 += [k, l]
						mag = [(a-b)**2 for a, b in zip(vec1, vec2)]
						mag = sum(mag)
						exp = math.exp(-0.5*mag/hc**2) + math.exp(-0.5*mag/hd**2)

						mag = math.sqrt(sum([a**2 for a in vec2]))
						meanSum += mag*exp
						total += exp

		if grayScale:
			newIm[i,j] = meanSum/total

	def meanShift(self, hc, hd, m=None, im=None, steps=10, grayScale=False):
		# Default using labd
		if im == None:
			im = self.labd

		if grayScale:
			im = self.rgb

		# Sets up info
		rows, cols, chans = im.shape
		ogIm = np.copy(im)
		ogIm = ogIm.astype(float)
		newIm = np.copy(im)
		newIm = newIm.astype(float)

		# Pool
		poolNum = 4
		p = Pool(poolNum)

		# Goes through pixels
		for step in tqdm(range(steps)):
			for i in tqdm(range(rows)):
				for j in range(cols):
					count, meanSum, total = 0, 0, 0

					if grayScale:
						x = [ogIm[i,j][0]]
						
					else:
						x = list(ogIm[i,j])

					# threadParmList = []
					# for k in range(rows):
					# 	for l in range(cols):
					# 		#def threadMS(self, x, rows, cols, ogIm, newIm, hc, hd, grayScale):
					# 		threadParmList.append([x, rows, cols, ogIm, newIm, hc, hd, grayScale])

					print(p.apply_async(self.threadMS, [x, rows, cols, ogIm, newIm, hc, hd, grayScale]))


					# for k in range(rows):
					# 	for l in range(cols):
					# 		if grayScale:
					# 			xi = [ogIm[k,l][0]]
					# 		else:
					# 			xi = list(ogIm[k,l])

					# 		if self.hfx(x, xi, hc) and self.hfx([i,j], [k,l], hd):
					# 			count += 1
								
					# 			if grayScale:
					# 				vec1, vec2 = x, xi
					# 				vec1 += [i, j] 
					# 				vec2 += [k, l]
					# 				mag = [(a-b)**2 for a, b in zip(vec1, vec2)]
					# 				mag = sum(mag)
					# 				exp = math.exp(-0.5*mag/hc**2) + math.exp(-0.5*mag/hd**2)

					# 				mag = math.sqrt(sum([a**2 for a in vec2]))
					# 				meanSum += mag*exp
					# 				total += exp

					# if grayScale:
					# 	newIm[i,j] = meanSum/total
					# 	# print(newIm[i,j])

			# Pool wait
			p.close()
			p.join()

			# One iter complete			
			temp = ogIm
			ogIm = newIm
			newIm = temp

		# Return
		self.meanshift = ogIm
		return True

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
	# # Test 1
	# test = imgRead('images/test-00.jpg')
	# xyz = convertXYZ(test)
	# labF, labD  = rgb2lab(test)
	# imWrite('output/test/xyz.png', xyz)
	# imWrite('output/test/lab.png', labD)

	# # Test 2
	# test = maFrignImg('images/test-01.jpg')
	# test.rgb2lab()
	# test.save('output/test/test-01-lab.png', test.labd)
	# test.save('output/test/test-01.png')

	# # Test 3
	# test = maFrignImg('images/test-01.jpg')
	# test.rgb2lab()
	# test.lab2rgb()
	# test.save('output/test/test-01-lab2rgb.png')

	# Test 4
	#def meanShift(self, hc, hd, m=None, im=None, steps=10, grayScale=False):
	test = maFrignImg('{}'.format(sys.argv[1]))
	test.meanShift(7, 8, 40, steps=5, grayScale=True)
	test.save('{}'.format(sys.argv[2]))

if __name__ == '__main__':
	DEBUG = True
	main()