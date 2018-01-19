import cv2
import os
import numpy as np    
import re                                                                                                 

'''
Function to obtain frames from a video
'''
def video_to_frames(dir, dirname, file):

	vidcap = cv2.VideoCapture(dir)

	'''
	Get length of the videos in frames:
		- Get frames from 50% of the video's length until 90% of the length
	'''
	length_of_video = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

	success, image = vidcap.read()


	new_dir = dir.replace('patches','frames3D').split('.MP4')[0]

	# print( new_dir )
	print( os.path.basename(new_dir) )

	count = 0
	frame_counter = 0
	vid = []
	while success:
		success,image = vidcap.read()



		if frame_counter > 16:
			frame_counter = 0
			print(file)
			# np.save(, vid)
			vid = []
		
		if count > int(length_of_video*.6) and count < int(length_of_video*.9):
			vid.append(dirname)
			# cv2.imwrite(os.path.join(dirname, file + 'frame%d.jpg' % count), image)     # save frame as JPEG file


		if cv2.waitKey(10) == 27:                     # exit if Escape is hit
			break

		count += 1
		frame_counter += 1


# vid = []
# while True:
#     ret, img = cap.read()
#     if not ret:
#         break
#     vid.append(cv2.resize(img, (171, 128)))
# vid = np.array(vid, dtype=np.float32)


'''
Function to obtain a list of files from a directory
'''
def list_files(dir):                                                                                                  
	r = []
	r_subdir = []
	r_file = []
	subdirs = [x[0] for x in os.walk(dir)]                                                                            
	for subdir in subdirs:                                                                                            
		files = os.walk(subdir).__next__()[2]                                                                             
		if (len(files) > 0):                                    
			for file in files:                                                                                        
				r.append( os.path.join(subdir, file) )   
				r_subdir.append(subdir)
				r_file.append(file)                
	return r, r_subdir, r_file


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='Generate the dataset for the 3D CNN')
	parser.add_argument('-dataset', type=str, default='patches', 
											help='Name of the dataset to use')
	parser.add_argument('-set', type=str, default='training',
											help='Set to use (training, validation, test)')
	args = parser.parse_args()

	'''
	@ r: list of directory path
	@ r_subdir: list of subdirectories
	@ r_file: list of files
	'''
	path_to_dataset = os.path.join('..', '2_Model', '2_Dataset', args.dataset, args.set)
	r, r_subdir, r_file = list_files(path_to_dataset)

	for video, dirname, file in zip(r, r_subdir, r_file):
		video_to_frames(video, dirname, file)