import numpy as np 
import imageio, os, math, sys
from tqdm import tqdm

def meanshift_gs(img_in, img_out, steps, hr, hs, M, sdr, sds, img_og=None):
	for step in tqdm(range(steps)):
		for i in range(img_in.shape[0]):
			for j in range(img_in.shape[1]):
				X = img_in[i, j]
				sumX = 0.0
				sumE = 0.0
				count = 0

				for k in range(img_in.shape[0]):
					for l in range(img_in.shape[1]):
						Xi = img_in[i, j]
						magr = abs(X - Xi)
						mags = math.sqrt(math.pow(i-k, 2) + math.pow(j-l, 2))

						if magr <= hr*sdr and mags <= hs*sds:
							count += 1
							exp = math.exp(-0.5 * (float(magr)**2 / hr**2 + float(mags)**2 / hs**2))
							sumX += Xi * exp
							sumE += exp

				if count >= M:
					img_out[i, j] = sumX / sumE

					# if abs(img_og[i,j] - img_out[i, j]) >= 5:
					# 	print(img_og[i,j], img_out[i, j])

		img_in = np.copy(img_out)
		# print()

	return img_out

def main(input_path='images/shapes_128.png', out_path='output/shapes_128_test.png', steps=5, hr=8, hs=7, M=10, sdr=3, sds=3, grayscale=True):
	# Opens image
	img_og = imageio.imread(input_path)
	img_in = np.copy(img_og)
	img_out = np.copy(img_og)

	if grayscale:
		img_out = meanshift_gs(img_in, img_out, steps, hr, hs, M, sdr, sds, img_og)
		imageio.imwrite(out_path, img_out)
	else:
		pass

if __name__ == '__main__':
	# input_path, out_path, steps, hr, hs, m, sdr, sds, grayscale = sys.argv[1:]
	# main(input_path, out_path, steps, hr, hs, m, sdr, sds, grayscale)
	main()