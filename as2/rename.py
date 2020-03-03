import os, re, sys

for root, dirs, files in os.walk(sys.argv[1]):

	reg = re.compile(r'^(?:frame)(?:_)?(\d+\.png|\d+\.jpg)$')

	if files:
		fileList = [f for f in files if f.endswith('.png') or f.endswith('.jpg')]
		newFileList = [reg.search(f).group(1) for f in fileList if reg.match(f)]
		newFileList = [str(int(f.split('.')[0])) + '.' + f.split('.')[1] for f in newFileList]

		for old, new in zip(fileList, newFileList):
			os.system('mv {} {}'.format(os.path.join(root, old), os.path.join(root, new)))