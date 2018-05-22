# export CUDA_VISIBLE_DEVICES=1,2
import os
import sys
import time
import os.path as osp

# Add the parent module to the import path
sys.path.append(osp.realpath(osp.join(osp.dirname(__file__), '../')))

import keras.backend as K

from keras.initializers import glorot_uniform
from keras.optimizers import RMSprop, SGD, adam
from keras.metrics import top_k_categorical_accuracy
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Dense, Activation, Flatten, Input
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, Callback

from collections import OrderedDict
from preprocessing.image import ImageDataGenerator

from _saving.SaveModelDirectory import create_path
from _saving.ModelParameters import ModelParameters

# import warnings
# warnings.simplefilter("ignore", DeprecationWarning)


__author__ = 'Daniel Garcia Zapata'

# Flow From Directory
def obtain_datagen(datagen, train_path, h5=True):
	return datagen.flow_from_directory(
				train_path,
				# target_size=(224,224),
				batch_size=batch_size,
				class_mode='binary',
				h5=h5)

# Yield for data generators
def generate_data_generator_for_two_images(genX1):
	while True:
		X1i = genX1.next()
		yield X1i[0], X1i[1]

if __name__ == '__main__':

	__model__ = 'CNN-LSTM_Features'

	print("Starting:", time.ctime())

	###########################################
	# Data

	input_shape = 4096

	dataset = 'SASE-FE'
	partition = 'prealigned'
	if dataset == 'OULU-CASIA':			# OULU-CASIA
		train_data_dir = os.path.join('..', '..', '_Dataset', dataset, 'consecutive', 'training')	
		validation_data_dir = os.path.join('..', '..', '_Dataset', dataset, 'consecutive', 'validation')
		n_output = 6
	else:
		train_data_dir = os.path.join('..', '..', '_Dataset', dataset, '5frames', 'training')
		validation_data_dir = os.path.join('..', '..', '_Dataset', dataset, '5frames', 'validation')
		n_output = 12

	###########################################
	# Parameters
	
	epochs = 20
	batch_size = 60
	nb_train_samples = 29798 / batch_size
	nb_validation_samples = 4428 / batch_size
	impl = 2 			# gpu

	############################################
	# Model

	neurons = 2622	# 512
	drop = 0.5
	lr = 0.0001
	nlayers = 3

	activation = 'relu'
	activation_r = 'sigmoid'

	# Initialize weights
	weight_init = glorot_uniform(seed=3)	

	'''
	Load the output of the CNN
	'''

	model = Sequential()
	if nlayers == 1:
		model.add(LSTM(neurons, input_shape=(None, input_shape), implementation=impl, dropout=drop,
						activation=activation, recurrent_activation=activation_r))
	else:
		model.add(LSTM(neurons, input_shape=(None, input_shape), implementation=impl, dropout=drop,
						activation=activation, recurrent_activation=activation_r, return_sequences=True))
		for i in range(1, nlayers-1):
			model.add(LSTM(neurons, dropout=drop, implementation=impl,
						  activation=activation, recurrent_activation=activation_r, return_sequences=True))
		model.add(LSTM(neurons, dropout=drop, implementation=impl,
					  activation=activation, recurrent_activation=activation_r))
 
	model.add(Dense(n_output, activation='softmax', kernel_initializer=weight_init))

	model.summary()

	###########################################
	# Data Generator

	datagen = ImageDataGenerator()

	# Training data generators
	train_generator = obtain_datagen(datagen, train_data_dir)
	validation_generator = obtain_datagen(datagen, validation_data_dir)	

	# Yield for data generators
	dataset_train_gen = generate_data_generator_for_two_images(train_generator)
	dataset_val_gen = generate_data_generator_for_two_images(validation_generator)

	###########################################
	''' Training '''

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
							'nlayers': nlayers,
							'date': time.ctime()})

	# Create Version folder
	export_path = create_path(__model__, dataset)

	checkpointer = ModelCheckpoint(filepath=osp.join(export_path, __model__+'_epoch-{epoch:02d}_val-accu-{val_acc:.4f}.hdf5'), verbose=1) #, save_best_only=True) 
	csv_logger = CSVLogger(osp.join(export_path, '_logs_'+__model__+'.log'), separator=',', append=False)
	tensorboard = TensorBoard(log_dir=export_path, histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
	model_parameters = ModelParameters(osp.join(export_path, '_model_'+__model__+'.log'), row_dict)

	'''
	Fit Model
	'''
	model.fit_generator(dataset_train_gen,
						steps_per_epoch=nb_train_samples,
						epochs=epochs,
						validation_data=dataset_val_gen,
						validation_steps=nb_validation_samples,
						callbacks=[checkpointer, csv_logger, tensorboard, model_parameters])

	print("Ending:", time.ctime())
