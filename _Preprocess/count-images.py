import os                                                                                             
import time
import numpy as np

from collections import defaultdict

'''
Function to obtain frames from a video
'''
def move_videos(classes, list_of_videos, dataset):

	new_base_path = os.path.join('..', '_Dataset', dataset, 'patch')

	for old_video in list_of_videos:
		participant_number = os.path.dirname(old_video).split('/')[6]

		# Validation
		if int(filter(str.isdigit, os.path.dirname(old_video).split('/')[6])) > 60:
			new_path = os.path.join(new_base_path, 'validation', classes)
			new_file_name = participant_number +'_'+ os.path.basename(old_video)

			# if not os.path.exists(new_path):
			# 	os.makedirs(new_path)

			full_path = os.path.join(new_path, new_file_name)
			# os.rename(old_video, full_path)

		# Training
		else:
			new_path = os.path.join(new_base_path, 'training', classes)
			new_file_name = participant_number +'_'+ os.path.basename(old_video)

			# if not os.path.exists(new_path):
			# 	os.makedirs(new_path)			

			full_path = os.path.join(new_path, new_file_name)
			# os.rename(old_video, full_path)

			print old_video

	# frames_per_participants = defaultdict(int)
	# for k, video in enumerate(list_of_videos):
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

'''
Function to obtain a list of files from a directory
'''
def list_files(dir):
	from collections import defaultdict

	videos_by_class = defaultdict(list)
	subdirs = [x[0] for x in os.walk(dir)]

	for subdir in sorted(subdirs):
		files = os.walk(subdir).next()[2]
		for file in sorted(files):
			if file.endswith('.jpeg'):
				file_name = os.path.join(subdir, file)
				videos_by_class[os.path.basename(subdir)].append( file_name )

	return videos_by_class


if __name__ == '__main__':
	print 'Starting:', time.ctime(), '\n' 

	import argparse

	parser = argparse.ArgumentParser(description='Change folder order for OULU-CASIA dataset')
	parser.add_argument('-dataset', 	type=str, default='OULU-CASIA', 
											help='Name of the dataset to use')
	parser.add_argument('-partition', 	type=str, default='OriginalImg', 
											help='Name of the data partition to use')	
	parser.add_argument('-process', 		type=str, default='VL',
											help='Set to use (NI_Acropped, VL_Acropped)')
	args = parser.parse_args()

	path_to_dataset = os.path.join('..', '_Dataset', args.dataset, args.partition, args.process)

	# OULU-CASIA has an extra folder depth
	if args.dataset == 'OULU-CASIA':
		subfolder = 'Strong'
		path_to_dataset = os.path.join(path_to_dataset, subfolder)

	videos_by_class = list_files(path_to_dataset)

	for classes, videos in videos_by_class.items():
		print videos
		# move_videos(classes, videos_by_class[classes], args.dataset)

	print '\nEnding:', time.ctime()