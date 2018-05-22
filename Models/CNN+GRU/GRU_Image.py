# export CUDA_VISIBLE_DEVICES=1,2
import os
import sys
import time
import os.path as osp

# Add the parent module to the import path
sys.path.append(osp.realpath(osp.join(osp.dirname(__file__), '../')))

from keras.optimizers import adam
from keras.initializers import glorot_uniform
from keras.models import Sequential, Model, load_model
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, Callback
from keras.layers import Dense, Activation, Flatten, Input, GRU, TimeDistributed

from collections import OrderedDict
from preprocessing.image_img import ImageDataGenerator

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

from collections import OrderedDict
from _saving.SaveModelDirectory import create_path
from _saving.ModelParameters import ModelParameters

__author__ = 'Daniel Garcia Zapata'

# Flow From Directory
def obtain_datagen(datagen, train_path, h5=True):
	return datagen.flow_from_directory(
				train_path,
				target_size=(img_width,img_height),
				batch_size=batch_size,
				class_mode='binary',
				partition=partition) 	

# Yield for data generators
def generate_data_generator_for_two_images(genX1):
	while True:
		X1i = genX1.next()
		yield X1i[0], X1i[1]

if __name__ == '__main__':

	__model__ = 'CNN_GRU_Image'

	print('Starting:', time.ctime(), '\n')


	###########################################
	# Parameters
	
	epochs = 20
	batch_size = 20
	impl = 2 			# gpu	

	###########################################
	# Data
	
	img_width, img_height, channels = 224, 224, 3 		# Resolution of inputs
	input_shape = 4096

	dataset = 'SASE-FE'
	partition = 'prealigned'
	if dataset == 'OULU-CASIA':
		train_data_dir = os.path.join('..', '..', '_Dataset', dataset, 'consecutive', 'training')	
		validation_data_dir = os.path.join('..', '..', '_Dataset', dataset, 'consecutive', 'validation')

		frames = 5
		n_output = 6

		nb_train_samples = 5941 / batch_size
		nb_validation_samples = 2025 / batch_size

	else:
		partition = ''

		train_data_dir = os.path.join('..', '..', '_Dataset', dataset, '5frames', 'training')	
		validation_data_dir = os.path.join('..', '..', '_Dataset', dataset, '5frames', 'validation')

		frames = 5
		n_output = 12

		nb_train_samples = 27971 / batch_size
		nb_validation_samples = 4173 / batch_size		

	############################################
	# Model

	neurons = 1024
	nlayers = 2
	dropout = 0.5

	activation = 'relu'
	activation_r = 'sigmoid'

	# Initialize weights
	weight_init = glorot_uniform(seed=3)	

	'''
	Load the output of the CNN
	'''
	cnn_model = load_model(os.path.join('weights', 'CNN_prealigned.hdf5'))

	model_input = Input(shape=(frames, img_width, img_height, channels), 
						name='seq_input')

	x = TimeDistributed(cnn_model)(model_input)
	x = TimeDistributed(Flatten())(x)
	x = GRU(neurons, dropout=dropout, name='gru_1')(x)
	out = Dense(n_output, kernel_initializer=weight_init, name='out')(x)

	model = Model(inputs=[model_input], outputs=out)

	model.summary()

	''' Freeze previous layers '''
	for layer in cnn_model.layers:
		layer.trainable = False			

	###########################################
	# Data Generator

	datagen = ImageDataGenerator(
		rescale=1. / 224,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True)

	# Training data generators
	train_generator = obtain_datagen(datagen, train_data_dir)
	validation_generator = obtain_datagen(datagen, validation_data_dir)

	# Yield for data generators
	dataset_train_gen = generate_data_generator_for_two_images(train_generator)
	dataset_val_gen = generate_data_generator_for_two_images(validation_generator)

	############################################
	''' Training '''
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
							'nlayers': nlayers,
							'date': time.ctime()})

	# Create Version folder
	export_path = create_path(__model__, dataset)

	checkpointer = ModelCheckpoint(filepath=osp.join(export_path, __model__+'_epoch-{epoch:02d}_val-accu-{val_acc:.4f}.hdf5'), verbose=1) #, save_best_only=True) 
	csv_logger = CSVLogger(osp.join(export_path, '_logs_'+__model__+'.log'), separator=',', append=False)
	tensorboard = TensorBoard(log_dir=export_path, histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
	model_parameters = ModelParameters(osp.join(export_path, '_model_'+__model__+'.log'), row_dict)

	model.fit_generator(dataset_train_gen,
						steps_per_epoch=nb_train_samples,
						epochs=epochs,
						validation_data=dataset_val_gen,
						validation_steps=nb_validation_samples)
						# ,callbacks=[checkpointer, csv_logger, model_parameters])

	print('\nEnding:', time.ctime())
