import imageio, os, sys, numpy as np, tqdm, argparse
from shapely.geometry import Point, Polygon

def projection_transform(img_src, img_tgt, pts_src, pts_tgt, H):
	# Sets up shapely polygon for testing if inside space
	poly_src, poly_tgt = Polygon(pts_src), Polygon(pts_tgt)

	# Gets min/max x and y values for target polygon
	min_xt, min_yt = pts_tgt[0]
	max_xt, max_yt = pts_tgt[0]
	min_xs, min_ys = pts_src[0]
	max_xs, max_ys = pts_src[0]
	for pt_s, pt_t in zip(pts_src, pts_tgt):
		# Gets point
		xs, ys = pt_s
		xt, yt = pt_t

		# Finds x min/max
		if min_xt > xt: min_xt = xt 
		elif max_xt < xt: max_xt = xt 
		if min_xs > xs: min_xs = xs
		elif max_xs < xs: max_xs = xs

		# Finds y min/max
		if min_yt > yt: min_yt = yt 
		elif max_yt < yt: max_yt = yt 
		if min_ys > ys: min_ys = ys 
		elif max_ys < ys: max_ys = ys

	# Goes through target polygon and replaces pixels
	for i in tqdm.tqdm(range(min_yt, max_yt)):
		for j in range(min_xt, max_xt):
			# Gets point and checks if in polygon
			ppt = Point(j, i)
			if ppt.within(poly_tgt):
				# Calcs estimated source point
				pt_e = np.matmul(H, np.vstack([j, i, 1]))
				xs, ys = int(pt_e[0, 0] / pt_e[2, 0]), int(pt_e[1, 0] / pt_e[2, 0])

				# Checks if estimated point is in source polygon and clamps if not
				if xs > max_xs: xs = max_xs
				elif xs < min_xs: xs = min_xs
				if ys > max_ys: ys = max_ys
				elif ys < min_ys: ys = min_ys
				
				# Copies over mapped source pixel to target
				img_tgt[i, j] = img_src[ys, xs]

	# Returns new target img
	return img_tgt

def calc_homography(pts_src, pts_tgt):
	# Sets up initial p
	p = np.vstack([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
	sum_r, sum_r_prev = 100000000000, 1000000000001

	# Iterates until low residual
	while sum_r < sum_r_prev:
		# Sets sums back to zero
		sum_r_prev = sum_r
		sum_A, sum_b, sum_r = np.zeros((8, 8), dtype=float), np.zeros((8, 1), dtype=float), 0.0

		# Goes through all the points
		for pt_src, pt_tgt in zip(pts_src, pts_tgt):
			# Gets source and target points
			xs, ys = pt_src
			xt, yt = pt_tgt

			# Reshapes p to H
			H = np.append(p, 1).reshape((3,3))

			# Gets estimated src point from target point
			pt_e = np.matmul(H, np.vstack([xt, yt, 1]))
			xe, ye, d = pt_e[0, 0] / pt_e[2, 0], pt_e[1, 0] / pt_e[2, 0], pt_e[2, 0]
			
			# Calcs residual
			r = np.vstack([xs - xe, ys - ye])
			sum_r += np.matmul(r.T, r)

			# Creates jacobian
			J = np.asarray([
				[xt, yt, 1, 0, 0, 0, -1.0*xt*xe, -1.0*yt*xe],
				[0, 0, 0, xt, yt, 1, -1.0*xt*ye, -1.0*yt*ye]
			]) / d

			# Calcs A and b matrices
			A = np.matmul(J.T, J)
			b = np.matmul(J.T, r)

			# Addes to running sums
			sum_A += A 
			sum_b += b

		# Calcs delta p and adds to p
		dp = np.matmul(np.linalg.inv(sum_A), sum_b)
		p += dp

	# Returns reshaped p as H matrix
	print(f'Final residual: {sum_r}')
	return np.append(p, 1).reshape((3, 3))

def parse_config(config_path):
	with open(config_path) as cf:
		# Gets image filepath and opens it
		img_path = cf.readline().rstrip()
		img = imageio.imread(img_path)

		# Reads points
		pts = []
		for line in cf:
			if line[0] == '#':
				continue
			x, y = line.rstrip().split()
			pts.append((int(x), int(y)))

		return img, pts

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--source', help='source config file path', type=str)
	parser.add_argument('-t', '--target', help='target config file path', type=str)
	parser.add_argument('-o', '--output', help='output file path', type=str)
	return parser.parse_args()

def main():
	# Parses args
	args = parse_args()

	# Parses Config files and gets imgs/pts
	img_src, pts_src = parse_config(args.source)
	img_tgt, pts_tgt = parse_config(args.target)

	# Calcs homography matrix
	H = calc_homography(pts_src, pts_tgt)

	# Applies projection transformation
	img_tgt = projection_transform(img_src, img_tgt, pts_src, pts_tgt, H)

	# Checks if output directory exists, creates if doesnt
	dir_path = args.output.split(os.sep)[:-1]
	dir_path = os.path.join(*dir_path)
	if not os.path.exists(dir_path):
		os.makedirs(dir_path)

	# Writes output image
	imageio.imwrite(args.output, img_tgt)

if __name__ == '__main__':
	main()