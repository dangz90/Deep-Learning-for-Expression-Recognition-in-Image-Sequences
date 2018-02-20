import os
import time
import h5py
import numpy as np

from keras.preprocessing import image
from keras.models import Model, load_model

__author__ = 'Daniel Garcia Zapata'
__model__ = 'Feature_Vector'

def create_feature_vector(dir, features):
	img_width, img_height = 224, 224
	subdirs = [x[0] for x in os.walk(dir)][1:] 

	for subdir in subdirs:
		files = next(os.walk(subdir))[2]	# os.walk(subdir).__next__()[2]
		if (len(files) > 0):
			for file in files:
				# Check if destination directory exists if not then create it
				destination = subdir.replace('frontalization', 'feature_vector')
				if not os.path.exists(destination):
					os.makedirs(destination)

				# Load image and transform it into feature vector
				image_path = os.path.join(subdir, file)
				img = image.load_img(image_path, target_size=(img_width, img_height))
				x = image.img_to_array(img)
				x = np.expand_dims(x, axis=0)

				images = np.vstack([x])

				feature_vec = features.predict(images)

				# Save feature vector in h5 format
				feature_vector_path = os.path.join(destination, file.replace('.jpg', '.h5'))

				print('Saving', feature_vector_path)
				hf = h5py.File(feature_vector_path, 'w')
				hf.create_dataset('data', data=feature_vec)
				hf.close()

if __name__ == '__main__':

	print("Starting:", time.ctime())

	###########################################
	# Parameters
	
	train_data_dir = os.path.join('..', '..', '_Dataset', 'frontalization', 'training/NoSymmetry')
	validation_data_dir = os.path.join('..', '..', '_Dataset', 'frontalization', 'validation/NoSymmetry')

	if not os.path.exists(validation_data_dir):
		os.makedirs(validation_data_dir)

	############################################
	# Model

	'''	Load the output of the CNN '''
	cnn_model = load_model('weights/Stream_1_Frontalization_epoch-09_val-accu-0.35.hdf5')
	cnn_output = cnn_model.get_layer('fc7').output

	''' Feature Vector '''
	features = Model(cnn_model.input, cnn_output)
	# create_feature_vector(train_data_dir, features)
	create_feature_vector(validation_data_dir, features)

	print("Ending:", time.ctime())