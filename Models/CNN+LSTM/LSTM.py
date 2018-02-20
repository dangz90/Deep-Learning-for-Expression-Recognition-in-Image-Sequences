import os
import time
import numpy as np

from pretrained_lstm import define_model

from keras.models import Sequential, Model, load_model
from keras.layers import LSTM
from keras.layers import Dense, Activation, Flatten, Input
from keras.optimizers import RMSprop, SGD, adam
from keras.callbacks import ModelCheckpoint, CSVLogger
# from keras.preprocessing.image import ImageDataGenerator
from preprocessing.image import ImageDataGenerator

__author__ = 'Daniel Garcia Zapata'

if __name__ == '__main__':

	__model__ = 'CNN-LSTM_1'

	print("Starting:", time.ctime())

	###########################################
	# Parameters
	
	epochs = 20
	batch_size = 60
	nb_train_samples = 29798 / batch_size
	nb_validation_samples = 4428 / batch_size
	impl = 2 			# gpu

	############################################
	# Model

	neurons = 512
	drop = 0.5
	lr = 0.0001

	# features_model = 'weights/Stream_1_Frontalization_epoch-09_val-accu-0.35.hdf5'
	# model = define_model(12, lr, drop, neurons, features_model)

	neurons = 512
	drop = 0.5
	nlayers = 4

	activation = 'relu'
	activation_r = 'sigmoid'

	'''
	Load the output of the CNN
	'''

	model = Sequential()
	if nlayers == 1:
		model.add(LSTM(neurons, input_shape=(None, 4096), implementation=impl, dropout=drop,
					  activation=activation, recurrent_activation=activation_r))
	else:
		model.add(LSTM(neurons, input_shape=(None, 4096), implementation=impl, dropout=drop,
					  activation=activation, recurrent_activation=activation_r, return_sequences=True))
		for i in range(1, nlayers-1):
			model.add(LSTM(neurons, dropout=drop, implementation=impl,
						  activation=activation, recurrent_activation=activation_r, return_sequences=True))
		model.add(LSTM(neurons, dropout=drop, implementation=impl,
					  activation=activation, recurrent_activation=activation_r))
	model.add(Dense(1))

	print('Neurons: ', neurons, 'Layers: ', nlayers, activation, activation_r)
	print('')

	###########################################
	# Data

	input_shape = 4096
	train_data_dir = os.path.join('..', '..', '_Dataset', 'dataset16_consecutive_3d', 'training')
	validation_data_dir = os.path.join('..', '..', '_Dataset', 'dataset16_consecutive_3d', 'validation')

	# Data Generator
	train_datagen = ImageDataGenerator()
	test_datagen = ImageDataGenerator()

	# Flow From Directory
	def obtain_datagen(datagen, train_path, seed_number=7, h5=True):
		return datagen.flow_from_directory(
					train_path,
					target_size=(224,224),
					# batch_size=batch_size,
					seed=seed_number,
					class_mode='binary',
					h5=h5) 		

	# Training data generators
	train_generator = obtain_datagen(train_datagen, train_data_dir)
	validation_generator = obtain_datagen(test_datagen, validation_data_dir)

	# Yield for data generators
	def generate_data_generator_for_two_images(genX1):
		while True:
			X1i = genX1.next()
			yield X1i[0], X1i[1]

	# Yield for data generators
	dataset_train_gen = generate_data_generator_for_two_images(train_generator)
	dataset_val_gen = generate_data_generator_for_two_images(validation_generator)

	############################################
	# Training

	optimizer = adam(lr=0.0001)
	model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

	'''
	Callbacks
	'''
	filepath = 'weights'
	checkpointer = ModelCheckpoint(filepath= os.path.join(filepath, '_epoch-{epoch:02d}_val-accu-{val_acc:.2f}.hdf5'), verbose=1) #, save_best_only=True)
	csv_logger = CSVLogger('logs/'+__model__+'.log', separator=',', append=False)

	model.fit_generator(dataset_train_gen,
						steps_per_epoch=nb_train_samples,
						epochs=epochs,
						validation_data=dataset_val_gen,
						validation_steps=nb_validation_samples,
						callbacks=[checkpointer, csv_logger])

	print("Ending:", time.ctime())