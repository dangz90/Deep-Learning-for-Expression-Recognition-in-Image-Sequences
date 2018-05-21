# export CUDA_VISIBLE_DEVICES=1,2
import os
import time
import csv, six

from vggface_model import VGGFace

from keras.optimizers import adam
from keras.models import  Model, load_model
from keras.layers import Dense, Activation, Flatten, Input, concatenate
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, Callback

from collections import OrderedDict
from SaveModelDirectory import create_path
from preprocessing.image import ImageDataGenerator


# Datagen
def obtain_datagen(datagen, path, h5=False, npy=False):
	return datagen.flow_from_directory(
				path,
				target_size=(img_height, img_width),
				batch_size=batch_size,
				class_mode='binary')

# Yield for data generators
def generate_data_generator_for_two_images(genX):
	while True:
		Xi = genX.next()
		print( Xi[0].shape )
		print( Xi[1].shape )

		yield [Xi[0], Xi[1]], Xi[2]

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
			self.keys = ['dataset', 'partition', 'loss', 'lr', 'date']

		if not self.writer:
					class CustomDialect(csv.excel):
						delimiter = self.sep

					self.writer = csv.DictWriter(self.csv_file,
												 fieldnames=['model'] + self.keys, dialect=CustomDialect)
					if self.append_header:
						self.writer.writeheader()

		row_dict = OrderedDict({'model': __model__, 
								'dataset': dataset,
								'partition': partition,
								'loss': loss,
								'lr': lr,
								'date': time.ctime()})

		self.writer.writerow(row_dict)
		self.csv_file.flush()        

		self.csv_file.close()
		self.writer = None   		

if __name__ == '__main__':

	__model__ = 'CNN_Geometry'

	print "Starting:", time.ctime()

	###########################################
	# Parameters

	epochs = 20
	batch_size = 40

	###########################################
	# Data

	img_width, img_height, channel = 224, 224, 3     # Resolution of inputs

	dataset = 'OULU-CASIA'
	partition = 'prealigned'
	# Dataset for Stream 1    
	train_data_dir = os.path.join('..', '..', '_Dataset', dataset, partition, 'test')
	validation_data_dir = os.path.join('..', '..', '_Dataset', dataset, partition, 'validation')

	if dataset == 'OULU-CASIA':
		nb_class = 6		
		
		nb_train_samples = 7754
		nb_validation_samples = 2625
	else:
		nb_class = 12

		nb_train_samples = 29798
		nb_validation_samples = 4428

	nb_train_samples = nb_train_samples / batch_size
	nb_validation_samples = nb_validation_samples / batch_size

	'''
	Load the model
	'''
	vgg_model = load_model(os.path.join('weights', 'CNN_prealigned.hdf5'))

	'''
	Customize the model
	'''
	# Add geometry input
	input_geo = Input(shape=(136,), name='input-geometry')
	# geo_layer = Flatten(name='flatten-geo')(input_geo)

	# Obtain frontalization's last layer
	last_layer = vgg_model.get_layer('pool5').output
	for layer in vgg_model.layers:
	   layer.name = layer.name + "-model_frontalization"
	front_layer = Flatten(name='flatten-front')(last_layer)

	# Concatenate geo and front layers and customize it
	x = concatenate([front_layer, input_geo])
	x = Dense(4096, activation='relu', name='fc6')(x)
	x = Dense(4096, activation='relu', name='fc7')(x)
	x = Dense(2622, activation='relu', name='fc8')(x)
	out = Dense(nb_class, activation='softmax', name='fc9')(x)

	model = Model(
		[vgg_model.input, input_geo], 
		out)

	'''
	Dataset Generators
	'''
	datagen = ImageDataGenerator(
		rescale=1. / 224,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True)

	# Training and Validation data generators
	train_generator = obtain_datagen(datagen, train_data_dir, npy=True)
	validation_generator = obtain_datagen(datagen, validation_data_dir, npy=True)

	# Yield for data generators
	dataset_train_gen = generate_data_generator_for_two_images(train_generator)
	dataset_val_gen = generate_data_generator_for_two_images(validation_generator)


	'''
	Fine-tune the model
	'''
	# Freeze previous layers
	for i, layer in enumerate(vgg_model.layers):
		layer.trainable = False 

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
	export_path = create_path(__model__, dataset)

	checkpointer = ModelCheckpoint(filepath=os.path.join(export_path, __model__+'_epoch-{epoch:02d}_val-accu-{val_acc:.4f}.hdf5'), verbose=1) #, save_best_only=True) 
	csv_logger = CSVLogger(os.path.join(export_path, '_logs_'+__model__+'.log'), separator=',', append=False)
	# tensorboard = TensorBoard(log_dir=export_path, histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
	# model_parameters = ModelParameters(os.path.join(export_path, '_model_'+__model__+'.log'))

	'''
	Train the model
	'''
	print('*************************************\nTraining \n*************************************')
	model.fit_generator(
		dataset_train_gen,
		steps_per_epoch=nb_train_samples,
		epochs=epochs,
		validation_data=dataset_val_gen,
		validation_steps=nb_validation_samples,
		callbacks=[checkpointer, csv_logger])
