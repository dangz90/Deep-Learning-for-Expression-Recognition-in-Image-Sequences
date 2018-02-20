import os                                                                                             
import time
import numpy as np

from collections import defaultdict

__author__ = 'Daniel Garcia Zapata'

'''
Function to obtain frames from a video
'''
def video_to_frames(classes, list_of_videos, dataset, data_set, number_of_frames):

	print('Class:', classes)

	vid = []
	frame_counter = 1
	participant_frames_number = 1
	original_participant_number = os.path.basename(list_of_videos[0]).split('_')[0]		# Number of the participant

	# Frames per participants
	frames_per_participants_path = os.path.join('..', '_Dataset', 'frames_per_participants', classes+'_frames_per_participants.npy')
	participant_frames = np.load(frames_per_participants_path).tolist()

	frames_per_participants = defaultdict(int)
	for k, video in enumerate(list_of_videos):
		current_participant_number = os.path.basename(video).split('_')[0]

		# Frames to use: frame_counter + number_of_frames 	Size of the frames
		if data_set == 'validation':
			current_frame = int(video.replace('.jpg','').replace('.mp4frame', '=').split('=')[1])
		else:
			current_frame = int( video.replace('.jpg.jpg','').replace('.MP4frame', '=').split('=')[1] )

		frame_limit = participant_frames[current_participant_number]

		if frame_counter > number_of_frames and original_participant_number == current_participant_number:
			'''
			Save
			'''
			folder_name = os.path.dirname(video).replace(dataset, str(number_of_frames)+'frames')
			participant_and_class = os.path.basename(video).split('.')[0]

			if not os.path.exists(folder_name):
				os.makedirs(folder_name)

			path_to_save = os.path.join( folder_name, participant_and_class + '_' + str(participant_frames_number) )
			np.save( path_to_save, vid )
			
			# Delete first video in order to keep always size 16 before appending the next video
			del vid[0]
			participant_frames_number += 1

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
			for file in sorted(files):
				videos_by_class[os.path.basename(subdir)].append( os.path.join(subdir, file) )
             
	return videos_by_class


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='Generate a dataset of frames')
	parser.add_argument('-dataset', 			
							type=str, default='patches', 
							help='Name of the dataset to use')
	parser.add_argument('-set', 				
							type=str, default='validation',
							help='Set to use (training, validation)')
	parser.add_argument('-number_of_frames', 	
							type=int, default=5,
							help='Number of frames to be used to create the dataset')	
	args = parser.parse_args()

	''' __main__  '''
	print("Starting:", time.ctime())

	# Obtain list of files in dataset directories
	path_to_dataset = os.path.join('..', '_Dataset', args.dataset, args.set)
	videos_by_class = list_files(path_to_dataset)

	# Create npy file with {number_of_frames} frames to be used as an input
	for classes, videos in videos_by_class.items():
		video_to_frames(classes, videos_by_class[classes], args.dataset, args.set, args.number_of_frames)

	print("Ending:", time.ctime())