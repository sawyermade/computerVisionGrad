import os, re

for root, dirs, files in os.walk('moon_frames/'):

	reg = re.compile(r'^(?:frame)([\d]+\.png)$')

	if files:
		fileList = [f for f in files if f.endswith('.png')]	
		newFileList = [reg.search(f).group(1) for f in fileList]

		for old, new in zip(fileList, newFileList):
			os.system('mv {} {}'.format(os.path.join(root, old), os.path.join(root, new)))