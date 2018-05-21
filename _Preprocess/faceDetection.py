import os
# import cv2
import time
import numpy as np

'''
Function to obtain a list of files from a directory
'''
def list_files(dir):
	from collections import defaultdict

	videos_by_class = defaultdict(list)
	subdirs = [x[0] for x in os.walk(dir) if len(x[0].split('/')) >= 8]

	for subdir in sorted(subdirs):
		files = os.walk(subdir).next()[2]
		for file in sorted(files):
			if file.endswith('.jpeg'):
				file_name = os.path.join(subdir, file)
				videos_by_class[os.path.basename(subdir)].append( file_name )

	return videos_by_class

def detect_face(list_of_files):
	# Obtain XML with Haar Features
	face_cascade = cv2.CascadeClassifier('C:\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')

	# Cycle through the files
	for subdir, file in zip(r_subdir, r_file):
		print(subdir +'\\'+ file)

		img = cv2.imread(subdir +'\\'+ file)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# Classifier for detecting faces
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)
		for (x,y,w,h) in faces:
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			# Create the face patch
			detected_color = img[y:y+h, x:x+w]
			detected_color = cv2.resize(detected_color, (224, 224))
			# Save patch
			split_subdir = subdir.split('\\')
			cv2.imwrite(os.path.join('patches\\'+split_subdir[1]+'\\'+split_subdir[2]+'\\', file + '.jpg'), detected_color)     # save frame as JPEG file

if __name__ == '__main__':
	print 'Starting:', time.ctime(), '\n' 

	import argparse

	parser = argparse.ArgumentParser(description='Change folder order for OULU-CASIA dataset')
	parser.add_argument('-dataset', 	type=str, default='OULU-CASIA', 
											help='Name of the dataset to use')
	parser.add_argument('-partition', 	type=str, default='OriginalImg', 
											help='Name of the data partition to use')	
	parser.add_argument('-process', 		type=str, default='NI',
											help='Set to use (NI_Acropped, VL_Acropped)')
	args = parser.parse_args()

	path_to_dataset = os.path.join('..', '_Dataset', args.dataset, args.partition, args.process)

	# OULU-CASIA has an extra folder depth
	if args.dataset == 'OULU-CASIA':
		subfolder = 'Strong'
		path_to_dataset = os.path.join(path_to_dataset, subfolder)

	# Get list of files in dataset
	list_of_files = list_files(path_to_dataset)

	for classes, file in list_of_files.items():
		detect_face(classes, list_of_files[classes], args.dataset)

	print '\nEnding:', time.ctime()