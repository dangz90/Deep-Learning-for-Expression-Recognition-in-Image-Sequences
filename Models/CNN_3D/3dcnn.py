# export CUDA_VISIBLE_DEVICES=1,2
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense, Input, concatenate
from keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.metrics import confusion_matrix
from cnn3d_model import get_model

'''
Model Parameters
'''
# dimensions of our images.
model_name = '3D_CNN'
img_width, img_height = 224, 224 # Resolution of inputs
train_samples = 29799
validation_samples = 4428
epochs = 20
batch_size = 40
nb_train_samples = 29799 / batch_size
nb_validation_samples = 4428 / batch_size

# Dataset for Stream 1
train_data_dir = '2_Dataset/dataset_3d/training/'
validation_data_dir = '2_Dataset/dataset_3d/validation/'


# Custom parameters
nb_class = 12
hidden_dim = 512


'''
Load the models
'''
model = get_model(
	summary=True)


'''
Dataset Generators
'''
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

# Datagen
def obtain_datagen(datagen, train_path, seed_number=7):
	return datagen.flow_from_directory(
				train_path,
				target_size=(img_height, img_width),
				batch_size=batch_size,
				seed=seed_number,
				class_mode='binary')

# Data Generators
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


'''
Fine-tune the model
'''
# Freeze previous layers
# for i, layer in enumerate(model.layers):
# 	layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


'''
Callbacks
'''
checkpointer = ModelCheckpoint(filepath='weights/'+model_name+'.epoch-{epoch:02d}_val-accu-{val_acc:.2f}.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('logs/'+model_name+'.log', separator=',', append=False)


print('END')

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
callbacks=[checkpointer, csv_logger])