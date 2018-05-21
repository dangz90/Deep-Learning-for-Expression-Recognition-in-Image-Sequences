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

def create_path(directory, subdirectory, classes, partition, new_file_name, extension):
	new_path = os.path.join(directory, subdirectory, partition, classes)
	full_path = os.path.join(new_path, new_file_name + extension)

	if not os.path.exists(new_path):
		os.makedirs(new_path)	

	return full_path


def read_pickle(subdir, file_path):
	with open(file_path, 'rb') as f:
		data = pickle.load(f)

		participant_number = os.path.splitext(os.path.basename(file_path))[0]

		previous_emotion = -1
		for emotion, geometry, face in zip(data['emos'], data['geoms'], data['faces']):
			# Check count of emotion
			if emotion != previous_emotion:
				count = 0
			else:
				count += 1
			previous_emotion = emotion

			# Create file name
			if int(filter(str.isdigit, participant_number)) > 65:
				partition = 'validation'
			else:
				partition = 'training'

			new_file_name = participant_number +'_'+ format(count, '02')

			# Geometry
			full_path_geometry = create_path(subdir, '_Geometry', classes[emotion], partition, new_file_name, '.npy')
			np.save(full_path_geometry, geometry)

			# Face
			full_path_face = create_path(subdir, '_Face', classes[emotion], partition, new_file_name, '.jpg')
			img = Image.fromarray(face, 'RGB')
			img.save(full_path_face)

'''
Function to obtain a list of files from a directory
'''
def list_files(dir, extension):
	from collections import defaultdict

	# Explore sub directories
	subdirs = [x[0] for x in os.walk(dir)] # if len(x[0].split(os.sep)) >= 6

	list_of_files = defaultdict(list)
	# Cycle through sub directories
	for subdir in sorted(subdirs):
		# Obtain files from sub directories
		files = os.walk(subdir).next()[2]
		# Cycle through files
		for e, file in enumerate(sorted(files)):
			print 'Participant', e

			if file.endswith(extension):
				file_path = os.path.join(subdir, file)

				# Obtain file class and put geometry and image in respective folder
				read_pickle(os.path.dirname(subdir), file_path)

	return list_of_files

if __name__ == '__main__':
	import argparse
	
	print 'Starting:', time.ctime(), '\n' 

	parser = argparse.ArgumentParser(description='Change folder order for OULU-CASIA dataset')
	parser.add_argument('-dataset', 	type=str, default='OULU-CASIA', 
											help='Name of the dataset to use')
	parser.add_argument('-preprocess', 	type=str, default='croppedfaces', 
											help='Name of the data process to use')	
	parser.add_argument('-process', 	type=str, default='',
											help='Set to use (NI_Acropped, VL_Acropped)')
	parser.add_argument('-depth', 		type=str, default='',
											help='Add if there is an extra depth')
	parser.add_argument('-extension', 	type=str, default='.jpeg',
											help='Add extension to extract')
	args = parser.parse_args()

	path_to_dataset = os.path.join('..', '_Dataset', args.dataset, args.preprocess, args.process, args.depth)
	print 'Using folder:', path_to_dataset, '\n'

	'''
	Get the list of files
	'''
	list_of_files = list_files(path_to_dataset, args.extension)

	print '\nEnding:', time.ctime()