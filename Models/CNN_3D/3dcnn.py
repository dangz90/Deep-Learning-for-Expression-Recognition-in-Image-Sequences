from theano import function, config, shared, tensor
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], tensor.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, tensor.Elemwise) and
              ('Gpu' not in type(x.op).__name__)
              for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')


# export CUDA_VISIBLE_DEVICES=1,2
import os
import time
import csv, six

# import theano
# theano.config.device = 'gpu:5'
# theano.config.floatX = 'float32'

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

from cnn3d_model import get_model
from sklearn.metrics import confusion_matrix

from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint, CSVLogger, Callback

from collections import OrderedDict
from SaveModelDirectory import create_path
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
		train_data_dir = os.path.join('..', '..', '_Dataset', dataset, 'consecutive', 'training')	
		validation_data_dir = os.path.join('..', '..', '_Dataset', dataset, 'consecutive', 'validation')

		frames = 5
		n_output = 6

		nb_train_samples = 6019 / batch_size
		nb_validation_samples = 1947 / batch_size

	else:
		partition = 'frontalization'
		train_data_dir = os.path.join('..', '..', '_Dataset', dataset, '5frames', 'training')	
		validation_data_dir = os.path.join('..', '..', '_Dataset', dataset, '5frames', 'validation')

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
	export_path = create_path(__model__, dataset)

	checkpointer = ModelCheckpoint(filepath=os.path.join(export_path, __model__+'_epoch-{epoch:02d}_val-accu-{val_acc:.4f}.hdf5'), verbose=1) #, save_best_only=True) 
	csv_logger = CSVLogger(os.path.join(export_path, '_logs_'+__model__+'.log'), separator=',', append=False)
	model_parameters = ModelParameters(os.path.join(export_path, '_model_'+__model__+'.log'))

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
