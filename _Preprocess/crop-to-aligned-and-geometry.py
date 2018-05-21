import os                                                                                             
import time
import pickle
import numpy as np

from PIL import Image
from collections import defaultdict

__author__ = 'Daniel Garcia Zapata'


classes = {	0:'Anger',
			1:'Disgust',
			2:'Fear',
			3:'Happiness',
			4:'Sadness',
			5:'Surprise'}

'''
Function to obtain frames from a video
'''
def move_videos(classes, list_of_files, dataset):

	for old_video in list_of_files:
		participant_number = os.path.dirname(old_video).split(os.sep)[4]

		# Validation
		if int(filter(str.isdigit, os.path.dirname(old_video).split(os.sep)[4])) > 60:
			new_path = os.path.join(new_base_path, 'validation', classes)
			new_file_name = participant_number +'_'+ os.path.basename(old_video)

			if not os.path.exists(new_path):
				os.makedirs(new_path)

			new_full_path = os.path.join(new_path, new_file_name)
			# os.rename(old_video, new_full_path)

		# Training
		else:
			new_path = os.path.join(new_base_path, 'training', classes)
			new_file_name = participant_number +'_'+ os.path.basename(old_video)

			if not os.path.exists(new_path):
				os.makedirs(new_path)			

			new_full_path = os.path.join(new_path, new_file_name)
			# os.rename(old_video, new_full_path)			

	# frames_per_participants = defaultdict(int)
	# for k, video in enumerate(list_of_files):
	# 	current_participant_number = os.path.basename(video).split('_')[0]

	# 	# Frames to use: frame_counter + 16 	Size of the frames of the CNN
	# 	current_frame = int(video.replace('.jpg','').replace('.mp4frame', '=').split('=')[1])
	# 	frame_limit = participant_frames[current_participant_number]

	# 	if frame_counter > number_of_frames and original_participant_number == current_participant_number:
	# 		'''
	# 		Save
	# 		'''
	# 		path_to_save = os.path.join( os.path.dirname(video).replace('2_Model','2_3DModel').replace(dataset, 'dataset'+number_of_frames+'_consecutive_3d') , os.path.basename(video).split('.')[0] + '_' + str(participant_frames_number) )
	# 		print(path_to_save)
	# 		# np.save( path_to_save, vid )
			
	# 		# Delete first video in order to keep always size 16 before appending the next video
	# 		# del vid[0]

	# 	if original_participant_number != current_participant_number or k == len(list_of_videos)-1:
	# 		'''
	# 		Restore counter
	# 		'''
	# 		vid = []
	# 		frame_counter = 1
	# 		participant_frames_number = 1
	# 		original_participant_number = current_participant_number
		
	# 	vid.append(os.path.basename(video))
	# 	frame_counter += 1

def create_path(directory, subdirectory, classes, new_file_name, extension):
	new_path = os.path.join(directory, subdirectory, classes)
	full_path = os.path.join(new_path, new_file_name + extension)

	if not os.path.exists(new_path):
		os.makedirs(new_path)	

	return full_path


def read_pickle(subdir, file_path):
	with open(file_path, 'rb') as f:
		data = pickle.load(f)

		participant_number = os.path.splitext(os.path.basename(file_path))[0]

		for e, (emotion, geometry, face) in enumerate(zip(data['emos'], data['geoms'], data['faces'])):
			# Create file name
			new_file_name = participant_number +'_'+ str(e)

			# Geometry
			full_path_geometry = create_path(subdir, '_geometry', classes[emotion], new_file_name, '.npy')
			# np.save(full_path_geometry, geometry)

			# Face
			full_path_face = create_path(subdir, '_face', classes[emotion], new_file_name, '.jpg')

			print full_path_face


			# img = Image.fromarray(face, 'RGB')
			# result.save(img)

'''
Function to obtain a list of files from a directory
'''
def list_files(dir, extension):
	from collections import defaultdict

	# Explore sub directories
	subdirs = [x[0] for x in os.walk(dir) if len(x[0].split(os.sep)) > len(dir.split(os.sep))]

	list_of_files = defaultdict(list)
	# Cycle through sub directories
	for subdir in sorted(subdirs):
		# Obtain files from sub directories
		files = os.walk(subdir).next()[2]
		# Cycle through files
		for e, file in enumerate(sorted(files)):

			if file.endswith(extension):
				file_path = os.path.join(subdir, file)

				print file_path

		# 		# Obtain file class and put geometry and image in respective folder
		# 		read_pickle(os.path.dirname(subdir), file_path)

		# 		# list_of_files[os.path.basename(subdir)].append( file_path )

	return list_of_files


if __name__ == '__main__':
	print 'Starting:', time.ctime(), '\n' 

	import argparse

	parser = argparse.ArgumentParser(description='Change folder order for OULU-CASIA dataset')
	parser.add_argument('-dataset', 	type=str, default='OULU-CASIA', 
											help='Name of the dataset to use')
	parser.add_argument('-partition', 	type=str, default='croppedfaces', 
											help='Name of the data partition to use')	
	parser.add_argument('-process', 	type=str, default='',
											help='Set to use (NI_Acropped, VL_Acropped)')
	parser.add_argument('-depth', 		type=str, default='',
											help='Add if there is an extra depth')
	parser.add_argument('-extension', 		type=str, default='.jpeg',
											help='Add extension to extract')
	args = parser.parse_args()

	path_to_dataset = os.path.join('..', '_Dataset', args.dataset, args.partition, args.process, args.depth)
	print 'Using folder:', path_to_dataset

	list_of_files = list_files(path_to_dataset, args.extension)
	# for classes, videos in list_of_files.items():
	# 	move_videos(classes, list_of_files[classes], args.dataset)

	print '\nEnding:', time.ctime()