# import theano
# theano.config.device = 'gpu:5'
# theano.config.floatX = 'float32'

# export CUDA_VISIBLE_DEVICES=1,2
import os
import sys
import time
import os.path as osp

# Add the parent module to the import path
sys.path.append(osp.realpath(osp.join(osp.dirname(__file__), '../')))

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

from cnn3d_model import get_model
from sklearn.metrics import confusion_matrix

from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint, CSVLogger, Callback

from collections import OrderedDict

from _saving.SaveModelDirectory import create_path
from _saving.ModelParameters import ModelParameters

from preprocessing.image import ImageDataGenerator


__author__ = 'Daniel Garcia Zapata'

# Datagen
def obtain_datagen(datagen, train_path):
	return datagen.flow_from_directory(
				train_path,
				target_size=(img_height, img_width),
				batch_size=batch_size,
				class_mode='binary',
				npy=True)

# Yield for data generators
def generate_data_generator_for_two_images(genX1):
	while True:
		X1i = genX1.next()
		yield X1i[0], X1i[1]

if __name__ == '__main__':

	__model__ = '3DCNN'

	print('Starting:', time.ctime(), '\n')

	###########################################
	# Parameters

	epochs = 20
	batch_size = 20

	###########################################
	# Data

	img_width, img_height, channels = 112, 112, 3 		# Resolution of inputs
	input_shape = 4096

	dataset = 'SASE-FE'				# OULU-CASIA, SASE-FE
	if dataset == 'OULU-CASIA':
		partition = 'prealigned'
		train_data_dir = osp.join('..', '..', '_Dataset', dataset, 'consecutive', 'training')	
		validation_data_dir = osp.join('..', '..', '_Dataset', dataset, 'consecutive', 'validation')

		frames = 5
		n_output = 6

		nb_train_samples = 6019 / batch_size
		nb_validation_samples = 1947 / batch_size

	else:
		partition = 'frontalization'
		train_data_dir = osp.join('..', '..', '_Dataset', dataset, '5frames', 'training')	
		validation_data_dir = osp.join('..', '..', '_Dataset', dataset, '5frames', 'validation')

		frames = 5
		n_output = 12

		nb_train_samples = 29799 / batch_size
		nb_validation_samples = 4428 / batch_size	


	# Custom parameters
	hidden_dim = 512


	'''
	Load the models
	'''
	model = get_model(
		frames=frames,
		summary=False)

	'''
	Dataset Generators
	'''
	datagen = ImageDataGenerator(
		rescale=1. / 224,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True)

	# Data Generators
	train_generator = obtain_datagen(datagen, train_data_dir)
	validation_generator = obtain_datagen(datagen, validation_data_dir)

	# Yield for data generators
	dataset_train_gen = generate_data_generator_for_two_images(train_generator)
	dataset_val_gen = generate_data_generator_for_two_images(validation_generator)


	'''
	Fine-tune the model
	'''
	# Freeze previous layers
	# for i, layer in enumerate(model.layers):
	# 	layer.trainable = False

	# Compile the model
	lr = 0.0001
	optimizer = adam(lr=lr)
	loss = 'sparse_categorical_crossentropy'	
	model.compile(	loss=loss,
					optimizer=optimizer,		
					metrics=['accuracy', 'top_k_categorical_accuracy'])

	'''
	Callbacks
	'''
	row_dict = OrderedDict({'model': __model__,
						'dataset': dataset,
						'partition': partition,
						'loss': loss,
						'lr': lr,
						'date': time.ctime()})

	# Create Version folder
	export_path = create_path(__model__, dataset)

	checkpointer = ModelCheckpoint(filepath=osp.join(export_path, __model__+'_epoch-{epoch:02d}_val-accu-{val_acc:.4f}.hdf5'), verbose=1) #, save_best_only=True) 
	csv_logger = CSVLogger(osp.join(export_path, '_logs_'+__model__+'.log'), separator=',', append=False)
	model_parameters = ModelParameters(osp.join(export_path, '_model_'+__model__+'.log'), row_dict)	

	'''
	Train the model
	'''
	print('*************************************\nTraining \n*************************************')
	model.fit_generator(dataset_train_gen,
						steps_per_epoch=nb_train_samples,
						epochs=epochs,
						validation_data=dataset_val_gen,
						validation_steps=nb_validation_samples,
						callbacks=[checkpointer, csv_logger, model_parameters])

print('\nEnding:"', time.ctime())
