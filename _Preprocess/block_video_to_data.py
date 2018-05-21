import os
import cv2
import time
import numpy as np    

'''
Function to obtain frames from a video
'''
def video_to_frames(dir):

    vidPath = '/path/foo/video.mp4'
    shotsPath = '/%d.avi' # output path (must be avi, otherwize choose other codecs)
    segRange = [(0,40),(50,100),(200,400)] # a list of starting/ending frame indices pairs

    cap = cv2.VideoCapture(dir)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = int(cv2.VideoWriter_fourcc('X','V','I','D')) # XVID codecs

    for idx,(begFidx,endFidx) in enumerate(segRange):
        writer = cv2.VideoWriter(shotsPath%idx,fourcc,fps,size)
        cap.set(cv2.CAP_PROP_POS_FRAMES,begFidx)
        ret = True # has frame returned
        while(cap.isOpened() and ret and writer.isOpened()):
            ret, frame = cap.read()
            frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
            if frame_number < endFidx:
                writer.write(frame)
            else:
                break
        writer.release()


'''
Function to obtain a list of files from a directory
'''
def split_dataset(data_path, path_to_save):
	if (not os.path.isdir(data_path)):
		assert False
	if (not os.path.isdir(path_to_save)):
		os.mkdir(path_to_save)

	'''
	Obtain Subdirs
	'''
	subdirs = [x[0] for x in os.walk(data_path)][1:]

	for subdir in sorted(subdirs):
		files = os.walk(subdir).__next__()[2]

		'''
		Obtain .MP4 file
		'''
		for f in sorted(files):
			if f.endswith('.MP4'):
				
				print('Reading', os.path.join(subdir, f))

				video_to_frames(os.path.join(subdir, f))

'''
Main: EGG
'''
if __name__ == "__main__":
	print('Starting:', time.ctime(), '\n')

	import argparse

	parser = argparse.ArgumentParser(description='Change folder order for OULU-CASIA dataset')
	parser.add_argument('-dataset', 	type=str, default='Blocked_raw', 
											help='Name of the dataset to use')
	parser.add_argument('-participant', type=str, default='', 
											help='Name of the participant')	
	args = parser.parse_args()

	path_to_dataset = os.path.join('..', '_Dataset', args.dataset, args.participant)
	path_to_save = os.path.join('..', '_Dataset', 'Blocked')

	"""
	Read video
	"""
	split_dataset(path_to_dataset, path_to_save)

	# for file in files:
	# 	save_as_numpy( path_to_dataset, file )
		# split_dataset(path_to_dataset, path_to_save)

	print('\nEnding:', time.ctime())
