def mean_shift(img):
	#NOTE: this is for greyscale only images for now


	rows,cols = get_image_size(img)

	#spatial distance and color distance values
	h = 8
	h_color_dist = 7

	#secondary "image" to hold intermediate values
	placeHolder = np.zeros((rows,cols,1))

	#number of iterations being performed, need to add stopping criteria for motion
	for t in tqdm(range(0,10)):

		#go through every pixel in the image
		for i in range(rows):
			for j in range(cols):

				# assign x as our current point we are looking at
				x = img[i,j]
				sum2 = 0
				sum3 = 0

				# go through all the pixels in the image to comapre our current point to
				for k in range(rows):
					for l in range(cols):

						# xi is a possible point we are comparing to
						xi = img[k,l]

						# if the spatial distance in xy space between our current point and 
						# xi is less than some h distance we continue
						if( math.sqrt((i-k)**2 + (j-l)**2) < h ):

							# if above is true then we compare the color distance
							# in this case its grescale so its a simple subtraction 
							if abs(x-xi) <= 2*h_color_dist:

								#if this xi pixel passes the two above criteria than
								# we use it to add to our sum using the guassian functions
								sum2 += xi * math.exp(-0.5*(x-xi)**2.0 / h_color_dist**2)
								sum3 += math.exp(-0.5*(x-xi)**2.0 / h_color_dist**2)

				#apply the shift value to the current pixel
				xNew = x + ((sum2 / sum3)-x)

				# store that new pixel value in our temporary image
				placeHolder[i,j] = xNew

		#after each iteration we copy over all of the new values to the original image
		img = np.copy(placeHolder)

		# we set temporary "image" to all 0's 
		placeHolder = np.zeros((rows,cols,1))

		# we repeate until we do all iterations or we meet some stopping criteria

	return img