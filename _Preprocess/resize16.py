import os
import cv2
import numpy as np
from shutil import copyfile

'''
Function to obtain frames from a video
'''
def video_to_frames(classes, list_of_videos, dataset):
	print(classes)

	for videos in list_of_videos:
		f = np.load(videos)
		images_list = [fi for fi in f]

		for images in images_list:
			# Obtain original image
			source_path = os.path.dirname(videos)
			source = os.path.join(source_path.replace('dataset16_consecutive_3d', 'patches'), images)

			# Read source image and resize to 112,112
			img = cv2.imread(source)
			img = cv2.resize(img, (112, 112))

			# Save modified image
			destination = os.path.join(source_path.replace('dataset16_consecutive_3d', 'patches_consecutive112'), images)
			cv2.imwrite(destination, img)     # save frame as JPEG file

'''
Function to obtain a list of files from a directory
'''
def list_files(dir):
	from collections import defaultdict

	videos_by_class = defaultdict(list)

	subdirs = [x[0] for x in os.walk(dir)]                                                                            
	for subdir in subdirs:                                                                                            
		files = os.walk(subdir).__next__()[2]                                                                             
		if (len(files) > 0):                                    
			for file in files:
				videos_by_class[os.path.basename(subdir)].append( os.path.join(subdir, file) )
			 
	return videos_by_class


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='Generate the dataset for the 3D CNN')
	parser.add_argument('-dataset', type=str, default='patches', 
											help='Name of the dataset to use')
	parser.add_argument('-set', 	type=str, default='validation',
											help='Set to use (training, validation, test)')
	args = parser.parse_args()

	args.dataset = 'dataset16_consecutive_3d'
	path_to_dataset = os.path.join('..', '2_3DModel', '2_Dataset', args.dataset, args.set)
	videos_by_class = list_files(path_to_dataset)

	for classes, videos in videos_by_class.items():
		video_to_frames(classes, videos_by_class[classes], args.dataset)

	print('\nEND')