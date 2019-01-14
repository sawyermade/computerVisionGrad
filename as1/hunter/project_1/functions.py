import imageio
import numpy as np
from tqdm import tqdm
import math
#-----------------------------------------------------------------------------------------------#
# read in an image to a numpy array and return it
def rgb_image_read(filename):
	img = imageio.imread(filename)
	img = img[:,:,:3]
	img = img.astype(float)
	return img
#-----------------------------------------------------------------------------------------------#
def grey_image_read(filename):
	img = imageio.imread(filename)
	img = img.astype(float)
	return img
#-----------------------------------------------------------------------------------------------#
# given an image and output file name this function saves it
def image_save(out_filename, img):
	img = img.astype('uint8')
	imageio.imwrite(out_filename, img)
#-----------------------------------------------------------------------------------------------#
# this function computes image #rows and #cols and returns it
def get_image_size(img):
	row = img.shape[0]
	col = img.shape[1]
	return row,col
#-----------------------------------------------------------------------------------------------#
# this function converts an rgb image to l*a*b
def rgb_lab_helper_one(currPixelVal):
	if currPixelVal > 0.04045:
		currPixelVal = ((currPixelVal + 0.055) / 1.055)**2.4
	else: 
		currPixelVal = currPixelVal / 12.92
	return currPixelVal * 100.0
#-----------------------------------------------------------------------------------------------#
# this is a helper function for converting the x,y,z values
def rgb_lab_helper_two(currPixelVal):
	if currPixelVal > 0.008856:
		currPixelVal = currPixelVal**(1.0/3.0)
	else: 
		currPixelVal = (7.787*currPixelVal) + (16.0/116.0)
	return currPixelVal
#-----------------------------------------------------------------------------------------------#
# this is the main rgb to lab function
def rgb_to_lab(img):

	xyzKernal = np.array([ [0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505] ])

	rows,cols = get_image_size(img)
	Xn, Yn, Zn = 95.0456, 100.0, 108.8754

	for i in tqdm(range(0,rows)):
		for j in range(0,cols):

			img[i,j] = img[i,j] / 255.0

			r = rgb_lab_helper_one(img[i,j,0])
			g = rgb_lab_helper_one(img[i,j,1])
			b = rgb_lab_helper_one(img[i,j,2])

			img[i,j] = r,g,b

			img[i,j] = np.matmul(xyzKernal,img[i,j])

			X, Y, Z = img[i,j]

			X = rgb_lab_helper_two(X/Xn)
			Y = rgb_lab_helper_two(Y/Yn)
			Z = rgb_lab_helper_two(Z/Zn)

			L = (116.0 * Y) - 16.0
			A = 500.0 * ( X - Y)
			B = 200.0 * ( Y - Z)

			img[i,j] = L, A, B

	return img
#-----------------------------------------------------------------------------------------------#
def lab_to_rgb_helper_one(currPixelVal):
	if ( currPixelVal**3.0  > 0.008856 ): 
		currPixelVal = currPixelVal**3
	else:                       
		currPixelVal = ( currPixelVal - 16.0 / 116.0 ) / 7.787
	return currPixelVal
#-----------------------------------------------------------------------------------------------#
def lab_to_rgb_helper_two(currPixelVal):
	if ( currPixelVal > 0.0031308 ):
	 currPixelVal = 1.055 * ( currPixelVal**( 1.0 / 2.4 ) ) - 0.055
	else:                     
		currPixelVal = 12.92 * currPixelVal
	return currPixelVal
#-----------------------------------------------------------------------------------------------#
def lab_to_rgb(img):
	xyzKernal = np.array([ [3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570] ])
	rows,cols = get_image_size(img)
	Xn, Yn, Zn = 95.0456, 100.0, 108.8754

	for i in tqdm(range(0,rows)):
		for j in range(0,cols):

			L,A,B = img[i,j]

			Y = ( L + 16.0 ) / 116.0
			X = A / 500.0 + Y
			Z = Y - B / 200.0

			Y = (lab_to_rgb_helper_one(Y) * Yn) / 100.0
			Z = (lab_to_rgb_helper_one(Z) * Zn) / 100.0
			X = (lab_to_rgb_helper_one(X) * Xn) / 100.0

			img[i,j] = X, Y, Z

			img[i,j] = np.matmul(xyzKernal,img[i,j])

			r,g,b = img[i,j]

			r = lab_to_rgb_helper_two(r) * 255
			g = lab_to_rgb_helper_two(g) * 255
			b = lab_to_rgb_helper_two(b) * 255

			img[i,j] = r,g,b

	return img
#-----------------------------------------------------------------------------------------------#
def mean_shift(img):

	rows,cols = get_image_size(img)

	h = 7

	placeHolder = np.zeros((rows,cols,1))

	print(img.shape)

	colorThresh = 20

	for t in tqdm(range(75)):

		for i in range(h//2,rows - h//2):
			for j in range(h//2,cols - h//2):

				x = img[i,j]

				sum1 = 0
				sum2 = 0
				sum3 = 0

				for k in range(i-h//2,i+h//2):
					for l in range(j-h//2,j+h//2):

						xi = img[k,l]
						if abs(x-xi) <= 40:
							sum1 += math.exp(-0.5*(x-xi)**2.0 / h**2)
							sum2 += xi * math.exp(-0.5*(x-xi)**2.0 / h**2)
							sum3 += math.exp(-0.5*(x-xi)**2.0 / h**2)

						
						#print(str(sum2) + " , " + str(sum3) + " , " + str(x))

				xNew = x + ((sum2 / sum3)-x)

				#print(xNew)

				placeHolder[i,j] = xNew
				# img[i,j] = xNew

		img = np.copy(placeHolder)

		placeHolder = np.zeros((rows,cols,1))

	return img










		













