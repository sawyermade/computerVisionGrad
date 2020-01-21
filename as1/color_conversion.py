import numpy as np

DEBUG = False

# Converts color space
def rgb2xyz(im):
	# Creates empty matrix
	rows, cols, channels = im.shape
	xyzIm = np.zeros((rows, cols, 3), dtype=float)

	# Converts all pixels to XYZ
	for i in range(rows):
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
	return xyzIm

def xyz2rgb(xyzIm):
	# Creates empty matrix
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
	return rgb

def xyz2lab(xyzIm):
	# Xn, Yn, Zn
	Xn, Yn, Zn = 0.950456*100, 1.0*100, 1.088754*100
	if DEBUG : print('Xn = {}, Yn = {}, Zn = {}'.format(Xn, Yn, Zn))

	# Creates Lab matrix
	rows, cols, _ = xyzIm.shape
	labImF = np.zeros((rows, cols, 3), dtype=float)
	labImD = np.zeros((rows, cols, 3), dtype=float)

	# Converts XYZ to Lab
	for i in range(rows):
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

			Ld = L*255/100
			ad = a+128
			bd = b+128

			labImF[i,j] = [L, a, b]
			labImD[i,j] = [Ld, ad, bd]

			if DEBUG and i == 0 and j == 0: print(labImF[i,j], labImD[i,j])

	# Returns Lab
	return labImD

def lab2xyz(labIm):
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

			L = L*100/255
			a = a-128
			b = b-128

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
	return xyzIm

def rgb2lab(rgb):
	return xyz2lab(rgb2xyz(rgb))

def lab2rgb(lab):
	return xyz2rgb(lab2xyz(lab))