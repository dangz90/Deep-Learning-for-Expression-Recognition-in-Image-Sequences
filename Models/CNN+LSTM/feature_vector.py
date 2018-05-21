import os
import time
import h5py
import numpy as np

from keras.preprocessing import image
from keras.models import Model, load_model

__author__ = 'Daniel Garcia Zapata'
__model__ = 'Feature_Vector'

def create_feature_vector(dir, features, processed):
	img_width, img_height = 224, 224
	subdirs = [x[0] for x in os.walk(dir)]
	
	for subdir in sorted(subdirs):
		files = next(os.walk(subdir))[2]	# os.walk(subdir).__next__()[2]

		for file in sorted(files):
			if file.endswith('.jpeg'):		
				
				# Check if destination directory exists if not then create it
				destination = subdir.replace(processed, 'feature_vector_'+processed)
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
				feature_vector_path = os.path.join(destination, file.replace('.jpeg', '.h5'))

				print('Saving', feature_vector_path)
				hf = h5py.File(feature_vector_path, 'w')
				hf.create_dataset('data', data=feature_vec)
				hf.close()

if __name__ == '__main__':

	print("Starting:", time.ctime())

	###########################################
	# Parameters
	
	dataset = 'OULU-CASIA'
	processed = 'prealigned'
	train_data_dir = os.path.join('..', '..', '_Dataset', dataset, processed, 'training')	
	validation_data_dir = os.path.join('..', '..', '_Dataset', dataset, processed, 'validation')
	data_dir = [train_data_dir, validation_data_dir]

	############################################
	# Model

	'''	Load the output of the CNN '''
	weights = 'CNN_patch_epoch-06_val-accu-0.34.hdf5'
	cnn_model = load_model(os.path.join('weights', weights))
	cnn_output = cnn_model.get_layer('fc7').output

	''' Feature Vector '''
	features = Model(cnn_model.input, cnn_output)
	for dirs in data_dir:
		create_feature_vector(dirs, features, processed)

	print("Ending:", time.ctime())