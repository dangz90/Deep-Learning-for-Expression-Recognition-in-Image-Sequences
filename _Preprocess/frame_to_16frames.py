import os                                                                                             
import cv2
import json
import numpy as np
from collections import defaultdict

'''
Function to obtain frames from a video
'''
def video_to_frames(classes, list_of_videos, dataset):

	print(classes)

	vid = []
	frame_counter = 1
	participant_frames_number = 1
	original_participant_number = os.path.basename(list_of_videos[0]).split('_')[0]

	# Frames per participants
	participant_frames = np.load(classes+'_frames_per_participants.npy').tolist()

	frames_per_participants = defaultdict(int)
	for k, video in enumerate(list_of_videos):
		current_participant_number = os.path.basename(video).split('_')[0]

		# Frames to use: frame_counter + 16 	Size of the frames of the CNN
		current_frame = int(video.replace('.jpg','').replace('.mp4frame', '=').split('=')[1])
		frame_limit = participant_frames[current_participant_number]

		if frame_counter > 16 and original_participant_number == current_participant_number:
			'''
			Save
			'''
			path_to_save = os.path.join( os.path.dirname(video).replace('2_Model','2_3DModel').replace(dataset, 'dataset16_consecutive_3d') , os.path.basename(video).split('.')[0] + '_' + str(participant_frames_number) )
			np.save( path_to_save, vid )
			
			# Delete first video in order to keep always size 16 before appending the next video
			del vid[0]

		if original_participant_number != current_participant_number or k == len(list_of_videos)-1:
			'''
			Restore counter
			'''
			vid = []
			frame_counter = 1
			participant_frames_number = 1
			original_participant_number = current_participant_number
		
		vid.append(os.path.basename(video))
		frame_counter += 1

	print('\n')

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


	path_to_dataset = os.path.join('..', '2_Model', '2_Dataset', args.dataset, args.set)
	videos_by_class = list_files(path_to_dataset)

	for classes, videos in videos_by_class.items():
		video_to_frames(classes, videos_by_class[classes], args.dataset)

	print('\n', 'END')