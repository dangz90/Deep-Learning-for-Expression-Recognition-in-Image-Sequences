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

		if len(images_list) == 16:
			source = videos
			destination = videos.replace('dataset_3d', 'dataset16_3d')

			# Copy File from dataset_3d to patches
			copyfile(source, destination)

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

	args.dataset = 'dataset_3d'
	path_to_dataset = os.path.join('..', '2_3DModel', '2_Dataset', args.dataset, args.set)
	videos_by_class = list_files(path_to_dataset)

	for classes, videos in videos_by_class.items():
		video_to_frames(classes, videos_by_class[classes], args.dataset)

	print('END')