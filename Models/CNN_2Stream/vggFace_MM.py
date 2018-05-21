# export CUDA_VISIBLE_DEVICES=1,2
import os
import time
import csv, six

from keras.optimizers import adam
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, CSVLogger, Callback
from keras.layers import Flatten, Dense, Input, concatenate

from collections import OrderedDict
from preprocessing.image import ImageDataGenerator
from SaveModelDirectory import create_path

__author__ = 'Daniel Garcia Zapata'


# Datagen
def obtain_datagen(datagen, path, h5=False, npy=False):
	return datagen.flow_from_directory(
				path,
				target_size=(img_height, img_width),
				batch_size=batch_size,
				class_mode='binary',
				h5=h5)

# Yield for data generators
def generate_data_generator_for_two_images(genX):
	while True:
		Xi = genX.next()
		yield [ Xi[0], Xi[1] ], Xi[2]	

# Class for writing the model parameters
class ModelParameters(Callback):
	def __init__(self, filename, separator=',', append=False):
		self.sep = separator
		self.filename = filename
		self.append = append
		self.writer = None
		self.keys = None
		self.append_header = True
		self.file_flags = 'b' if six.PY2 and os.name == 'nt' else ''
		super(ModelParameters, self).__init__()

	def on_train_begin(self, logs=None):
		if self.append:
			if os.path.exists(self.filename):
				with open(self.filename, 'r' + self.file_flags) as f:
					self.append_header = not bool(len(f.readline()))
			self.csv_file = open(self.filename, 'a' + self.file_flags)
		else:
			self.csv_file = open(self.filename, 'w' + self.file_flags)

		if self.keys is None:
			self.keys = ['dataset', 'loss', 'lr', 'date']

		if not self.writer:
					class CustomDialect(csv.excel):
						delimiter = self.sep

					self.writer = csv.DictWriter(self.csv_file,
												 fieldnames=['model'] + self.keys, dialect=CustomDialect)
					if self.append_header:
						self.writer.writeheader()

		row_dict = OrderedDict({'model': __model__, 
								'dataset': dataset,
								'loss': loss,
								'lr': lr,
								'date': time.ctime()})

		self.writer.writerow(row_dict)
		self.csv_file.flush()        

		self.csv_file.close()
		self.writer = None 		

if __name__ == '__main__':

	__model__ = 'CNN_s1-face_s2-aligned'

	print "Starting:", time.ctime()

	###########################################
	# Parameters

	epochs = 20
	batch_size = 20

	###########################################
	# Data

	img_width, img_height, channel = 224, 224, 3     # Resolution of inputs
	nb_class = 6

	dataset = 'OULU-CASIA'
	# Dataset for Stream 1    
	train_data_dir = os.path.join('..', '..', '_Dataset', dataset, 'preface', 'training')
	validation_data_dir = os.path.join('..', '..', '_Dataset', dataset, 'preface', 'validation')


	if dataset == 'OULU-CASIA':
		nb_train_samples = 8384
		nb_validation_samples = 1982
	else:
		nb_train_samples = 29798
		nb_validation_samples = 4428

	nb_train_samples = nb_train_samples / batch_size
	nb_validation_samples = nb_validation_samples / batch_size	

	###########################################
	#	Create Models

	vgg_model_stream_1 = load_model(os.path.join('weights', 'CNN_patch_epoch-06_val-accu-0.34.hdf5'))
	vgg_model_stream_2 = load_model(os.path.join('weights', 'CNN_aligned_epoch-19_val-accu-0.2000.hdf5'))

	'''
	Customize the model
	'''
	stream_1 = vgg_model_stream_1.get_layer('pool5').output
	stream_1 = Flatten(name='flatten-1')(stream_1)

	for layer in vgg_model_stream_1.layers:
	   layer.name = layer.name + "-model_stream_1"

	stream_2 = vgg_model_stream_2.get_layer('pool5').output
	stream_2 = Flatten(name='flatten-2')(stream_2)

	for layer in vgg_model_stream_1.layers:
	   layer.name = layer.name + "-model_stream_2"

	# fuse_layer = stream_1 + stream_2
	fuse_layer = concatenate([stream_1, stream_2])

	fuse_layer = Dense(1024, activation='relu', name='fc6')(fuse_layer)
	fuse_layer = Dense(1024, activation='relu', name='fc7')(fuse_layer)
	fuse_layer = Dense(512, activation='relu', name='fc8')(fuse_layer)
	out = Dense(nb_class, activation='softmax', name='fc9')(fuse_layer)

	model = Model(
		[vgg_model_stream_1.input, vgg_model_stream_2.input],
		out)


	###########################################
	#	Dataset Generators

	datagen = ImageDataGenerator(
		rescale=1. / 224,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True)

	# Data generators
	train_generator = obtain_datagen(datagen, train_data_dir, h5=True)
	validation_generator = obtain_datagen(datagen, validation_data_dir, h5=True)

	# Yield for data generators
	dataset_train_gen = generate_data_generator_for_two_images(train_generator)
	dataset_val_gen = generate_data_generator_for_two_images(validation_generator)

	###########################################
	# 	Train the model
	
	# Freeze previous layers
	for i, layer in enumerate(vgg_model_stream_1.layers):
		layer.trainable = False
	for i, layer in enumerate(vgg_model_stream_2.layers):
		layer.trainable = False 

	# Compile the model
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
	# Create Version folder
	export_path = create_path(__model__)

	checkpointer = ModelCheckpoint(filepath= os.path.join(export_path, 'weights', __model__+'_epoch-{epoch:02d}_val-accu-{val_acc:.4f}.hdf5'), verbose=1) #, save_best_only=True) 
	csv_logger = CSVLogger(os.path.join(export_path, 'logs',__model__+'.log'), separator=',', append=False)
	model_parameters = ModelParameters(os.path.join(export_path, '_model_'+__model__+'.log'))

	'''
	Train the model
	'''
	print('*************************************\nFine-tuning the model \n*************************************')

	model.fit_generator(
		dataset_train_gen,
		steps_per_epoch=nb_train_samples,
		epochs=epochs,
		validation_data=dataset_val_gen,
		validation_steps=nb_validation_samples,
		callbacks=[checkpointer, csv_logger, model_parameters])

	print("Ending:", time.ctime())
