# export CUDA_VISIBLE_DEVICES=1,2
import os
import h5py
import time
import csv, six
import numpy as np
from keras import backend
from keras.models import load_model
from sklearn.metrics import confusion_matrix

try:
	from PIL import Image as pil_image
except ImportError:
	pil_image = None

classes = {0:'Anger', 1:'Disgust', 2:'Fear', 3:'Happiness', 4:'Sadness', 5:'Surprise'}
emotions = {'Anger':0, 'Disgust':1, 'Fear':2, 'Happiness':3, 'Sadness':4, 'Surprise':5}
# labels = ['Anger','Disgust','Fear','Happiness','Sadness','Surprise']
labels = [0, 1, 2, 3, 4, 5]


def load_img(path, grayscale=False, target_size=None,
			 interpolation='bilinear'):
	if pil_image is None:
		raise ImportError('Could not import PIL.Image. '
						  'The use of `array_to_img` requires PIL.')
	img = pil_image.open(path)
	if grayscale:
		if img.mode != 'L':
			img = img.convert('L')
	else:
		if img.mode != 'RGB':
			img = img.convert('RGB')
	if target_size is not None:
		width_height_tuple = (target_size[1], target_size[0])
		if img.size != width_height_tuple:
			if interpolation not in _PIL_INTERPOLATION_METHODS:
				raise ValueError(
					'Invalid interpolation method {} specified. Supported '
					'methods are {}'.format(
						interpolation,
						", ".join(_PIL_INTERPOLATION_METHODS.keys())))
			resample = _PIL_INTERPOLATION_METHODS[interpolation]
			img = img.resize(width_height_tuple, resample)
	return img  

def img_to_array(img, data_format=None):
	if data_format is None:
		data_format = backend.image_data_format()
	if data_format not in {'channels_first', 'channels_last'}:
		raise ValueError('Unknown data_format: ', data_format)
	# Numpy array x has format (height, width, channel)
	# or (channel, height, width)
	# but original PIL image has format (width, height, channel)
	x = np.asarray(img, dtype=backend.floatx())
	if len(x.shape) == 3:
		if data_format == 'channels_first':
			x = x.transpose(2, 0, 1)
	elif len(x.shape) == 2:
		if data_format == 'channels_first':
			x = x.reshape((1, x.shape[0], x.shape[1]))
		else:
			x = x.reshape((x.shape[0], x.shape[1], 1))
	else:
		raise ValueError('Unsupported image shape: ', x.shape)
	return x            

if __name__ == '__main__':

	__model__ = 'CNN_Geometry'

	print "Starting:", time.ctime()

	###########################################
	# Get Dataset
	dataset = 'OULU-CASIA'
	partition = 'aligned'
	directory = os.path.join('..', '..', '_Dataset', dataset, partition, '_Face', 'validation')

	def list_files(dir, extension):
		from collections import defaultdict

		# Explore sub directories
		subdirs = [x[0] for x in os.walk(dir)] # if len(x[0].split(os.sep)) >= 6

		names_of_files = defaultdict(list)
		list_of_img = defaultdict(list)
		list_of_geo = defaultdict(list)	
		# Cycle through sub directories
		for subdir in sorted(subdirs[1:]):

			# Obtain files from sub directories
			files = os.walk(subdir).next()[2]

			# Cycle through files
			for e, file in enumerate(sorted(files)):

				if file.endswith(extension):
					file_path = os.path.join(subdir, file)
					names_of_files[os.path.basename(subdir)].append( file )	

					''' Images '''
					grayscale = 'grayscale'
					data_format = backend.image_data_format()   
					image_shape = (224,224,3)
					index_array = np.arange(1)

					batch_x = np.zeros( (len(index_array),) + image_shape, dtype=backend.floatx() )	
					img = load_img( os.path.join(file_path),
									grayscale=grayscale,
									target_size=[224,224])
					x = img_to_array(img, data_format=data_format)
					batch_x[0] = x

					list_of_img[os.path.basename(subdir)].append( batch_x )	

					''' Geometry '''
					batch_x_2 = np.zeros( (len(index_array),) + (136,), dtype=backend.floatx() )

					# f = h5py.File(os.path.join(file_path.replace('prealigned', 'pregeometry').replace(extension, '.h5')), 'r')
					# x = f['geometry'].value
					f = np.load( os.path.join(file_path.replace('_Face', '_Geometry').replace(extension, '.npy')) )
					x = f

					# # Normalize
					new_origin = [112,135]
					new_x = x - new_origin
					new_x = (new_x - new_x.min())/(new_x.max() - new_x.min())
					x = new_x.reshape((136,))
					batch_x_2[0] = x 	

					list_of_geo[os.path.basename(subdir)].append( batch_x_2 )									

		return list_of_img, list_of_geo, names_of_files
		# return list_of_img, names_of_files

	extension = '.jpg'
	# list_of_img, list_of_geo, names_of_files = list_files(directory, extension)
	list_of_img, list_of_geo, names_of_files = list_files(directory, extension)

	'''
	Load the model
	'''
	model = load_model(os.path.join('weights', 'CNN_pregeometry.hdf5'))	

	print('START')

	y_true = []
	y_pred = []
	accuracy = 0.0
	total_files = 0

	for (class0, files), (class1, videos), (class2, geos) in zip(names_of_files.items(), list_of_img.items(), list_of_geo.items()):

		for file, video, geo in zip(files, videos, geos):

			inputs = [video, geo]
			# inputs = video

			prediction = model.predict( inputs )
			class_number = (np.argmax(prediction))

			y_true.append(emotions[class1])
			y_pred.append(class_number)

			# print(file, 'True', class1, 'Pred', classes[class_number])

			if class_number == emotions[class1]:
				accuracy += 1.0

			total_files += 1

	filename = 'confusion_matrix.txt'
	with open(filename, 'w') as f:
		f.write(np.array2string(confusion_matrix(y_true, y_pred, labels=labels), separator=', '))

	print('Accuracy', float(accuracy)/float(total_files) )

	print('END')
