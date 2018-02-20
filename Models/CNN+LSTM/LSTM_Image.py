import os
import sys
import time
import numpy as np

import keras.backend as K
from keras import initializers
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, TimeDistributed
from keras.layers import Dense, Activation, Flatten, Input
from keras.optimizers import RMSprop, SGD, adam
from keras.callbacks import ModelCheckpoint, CSVLogger

from SaveModelDirectory import create_path
from preprocessing.image_img import ImageDataGenerator


__author__ = 'Daniel Garcia Zapata'


def euclidean_distance_loss(y_true, y_pred):
	"""
	Euclidean distance loss
	https://en.wikipedia.org/wiki/Euclidean_distance
	:param y_true: TensorFlow/Theano tensor
	:param y_pred: TensorFlow/Theano tensor of the same shape as y_true
	:return: float
	"""
	return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

if __name__ == '__main__':

	__model__ = 'CNN-LSTM_Image'

	print("Starting:", time.ctime())

	###########################################
	# Data

	input_shape = 4096
	train_data_dir = os.path.join('..', '..', '_Dataset', '5frames', 'training')
	validation_data_dir = os.path.join('..', '..', '_Dataset', '5frames', 'validation')	

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
	dropout = 0.5
	nlayers = 2
	n_output = 12

	activation = 'relu'
	activation_r = 'sigmoid'

	# Initialize weights
	weight_init = initializers.glorot_uniform(seed=3)	

	'''
	Load the output of the CNN
	'''

	cnn_model = load_model('weights/Stream_1_Frontalization_epoch-09_val-accu-0.35.hdf5')

	model_input = Input(shape=(None,224,224,3), name='seq_input')

	# little hack to remove the last layers after the last cnn layer, so we remove them
	# using pop
	cnn_model.layers.pop()
	cnn_model.layers.pop()
	cnn_model.layers.pop()
	cnn_model.layers.pop()
	cnn_model.layers.pop()
	cnn_model.layers.pop()
	cnn_model.layers[-1].outbound_nodes = []
	cnn_model.outputs = [cnn_model.layers[-1].output]

	for layer in cnn_model.layers:
		layer.trainable = False

	x = TimeDistributed(Flatten())(model_input)
	x = LSTM(neurons, dropout=dropout, name='lstm')(x)
	out = Dense(n_output, kernel_initializer=weight_init, name='out')(x)

	model = Model(inputs=[model_input], outputs=out)

	model.summary()

	###########################################
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
	model.compile(	loss=euclidean_distance_loss,
					optimizer=optimizer,		
					metrics=['accuracy'])

	'''
	Callbacks
	'''
	
	# Create Version folder
	export_path = create_path(__model__)

	checkpointer = ModelCheckpoint(filepath= os.path.join(export_path, 'weights', __model__+'_epoch-{epoch:02d}_val-accu-{val_acc:.2f}.hdf5'), verbose=1) #, save_best_only=True) 
	csv_logger = CSVLogger(os.path.join(export_path, 'logs',__model__+'.log'), separator=',', append=False)

	model.fit_generator(dataset_train_gen,
						steps_per_epoch=nb_train_samples,
						epochs=epochs,
						validation_data=dataset_val_gen,
						validation_steps=nb_validation_samples,
						callbacks=[checkpointer, csv_logger])

	print("Ending:", time.ctime())