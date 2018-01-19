# export CUDA_VISIBLE_DEVICES=1,2
from keras.models import  Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense, Input
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from vggface_model import VGGFace
import numpy

'''
Model Parameters
'''
# dimensions of our images.
model_name = 'Stream_1_Frontalization'
img_width, img_height = 224, 224 # Resolution of inputs
channel = 3
train_data_dir = '2_Dataset/frontalization/training/NoSymmetry/'
validation_data_dir = '2_Dataset/frontalization/validation/NoSymmetry'
nb_train_samples = 29798
nb_validation_samples = 4428
epochs = 20
batch_size = 60

nb_train_samples = nb_train_samples / batch_size
nb_validation_samples = nb_validation_samples / batch_size

#custom parameters
nb_class = 12
hidden_dim = 512

'''
Load the model
'''
vgg_model = VGGFace(
	include_top=False,
	input_shape=(img_width, img_height, 3))

'''
Custom the model
'''
last_layer = vgg_model.get_layer('pool5').output
x = Flatten(name='flatten')(last_layer)
x = Dense(hidden_dim, activation='relu', name='fc6')(x)
x = Dense(hidden_dim, activation='relu', name='fc7')(x)
x = Dense(hidden_dim, activation='relu', name='fc8')(x)
out = Dense(nb_class, activation='softmax', name='fc9')(x)

custom_vgg_model = Model(
	vgg_model.input, 
	out)

'''
Dataset Generators
'''
train_datagen = ImageDataGenerator(
    rescale=1. / 224,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(
	rescale=1. / 224)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

'''
Fine-tune the model
'''
# Freeze previous layers
for i, layer in enumerate(vgg_model.layers):
	layer.trainable = False 

# Compile the model
custom_vgg_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

'''
Callbacks
'''
checkpointer = ModelCheckpoint(filepath='weights/'+model_name+'_epoch-{epoch:02d}_val-accu-{val_acc:.2f}.hdf5', verbose=1) #, save_best_only=True)
csv_logger = CSVLogger('logs/'+model_name+'.log', separator=',', append=False)

'''
Train the model
'''
print('*************************************\nFine-tuning the model \n*************************************')
custom_vgg_model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples,
    callbacks=[checkpointer, csv_logger])