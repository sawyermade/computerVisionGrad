import numpy as np 
import imageio, os, math, sys
from tqdm import tqdm
from color_conversion import *

def meanshift_gs(img_og, steps, hr, hs, M, sdr, sds):
	img_in = np.copy(img_og)
	img_out = np.copy(img_og)
	for step in tqdm(range(steps)):
		for i in range(img_in.shape[0]):
			for j in range(img_in.shape[1]):
				X = int(img_in[i, j])
				sumX = 0.0
				sumE = 0.0
				count = 0

				for k in range(img_in.shape[0]):
					for l in range(img_in.shape[1]):
						Xi = int(img_in[k, l])
						magr = abs(X - Xi)
						mags = math.sqrt(math.pow(i-k, 2) + math.pow(j-l, 2))

						if magr <= hr*sdr and mags <= hs*sds:
							count += 1
							exp = math.exp(-0.5 * (magr**2 / hr**2 + mags**2 / hs**2))
							sumX += Xi * exp
							sumE += exp

				if count >= M:
					img_out[i, j] = sumX // sumE

		img_in = np.copy(img_out)

	return img_out

def meanshift_color(img_og, steps, hr, hs, M, sdr, sds):
	img_in = np.copy(img_og)
	img_out = np.copy(img_og)
	for step in tqdm(range(steps)):
		for i in range(img_in.shape[0]):
			for j in range(img_in.shape[1]):
				X = img_in[i, j].astype(np.float32)
				sumX = np.zeros((3), dtype=np.float32)
				sumE = 0.0
				count = 0

				for k in range(img_in.shape[0]):
					for l in range(img_in.shape[1]):
						Xi = img_in[k, l].astype(np.float32)
						magr = math.sqrt(np.sum((X - Xi)**2))
						mags = math.sqrt(math.pow(i-k, 2) + math.pow(j-l, 2))

						if magr <= hr*sdr and mags <= hs*sds:
							count += 1
							exp = math.exp(-0.5 * (magr**2 / hr**2 + mags**2 / hs**2))
							sumX += Xi * exp
							sumE += exp

				if count >= M:
					img_out[i, j] = sumX / sumE

		img_in = np.copy(img_out)

	return img_out

def main(input_path='images/baboon_128.png', out_path='output/baboon_128_output_40.png', steps=5, hr=8, hs=7, M=40, sdr=3, sds=3, grayscale=False):
	# Opens image
	img_og = imageio.imread(input_path)

	if grayscale:
		img_out = meanshift_gs(img_og, steps, hr, hs, M, sdr, sds)
		
	else:
		img_lab = rgb2lab(img_og)
		img_out = meanshift_color(img_lab, steps, hr, hs, M, sdr, sds)
		img_out = lab2rgb(img_out)

	imageio.imwrite(out_path, img_out.astype(np.uint8))

if __name__ == '__main__':
	# input_path, out_path, steps, hr, hs, m, sdr, sds, grayscale = sys.argv[1:]
	# main(input_path, out_path, steps, hr, hs, m, sdr, sds, grayscale)
	main()