# export CUDA_VISIBLE_DEVICES=1,2
import h5py
import numpy as np
from vggface_model import VGGFace
from keras.models import  Model, load_model
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
# from keras.preprocessing.image import ImageDataGenerator
from preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense, Input, concatenate
from obtain_geometry_dataset import obtain_geometry_batch


'''
Model Parameters
'''
# dimensions of the images
model_name = 'PreFrontalization_Geometry_Centered_Normalize'
img_width, img_height = 224, 224    # Resolution of inputs
channel = 3
nb_train_samples = 29798
nb_validation_samples = 4428
epochs = 20
batch_size = 60

nb_train_samples = 29798 / batch_size
nb_validation_samples = 4428 / batch_size

# Dataset for Stream 1
train_data_dir = '2_Dataset/frontalization/training/NoSymmetry/'
validation_data_dir = '2_Dataset/frontalization/validation/NoSymmetry/'

train_data_dir_2 = '2_Dataset/geometry/training/'
validation_data_dir_2 = '2_Dataset/geometry/validation/'

# custom parameters
nb_class = 12
hidden_dim = 512


'''
Load the model
'''
vgg_model_frontalization = load_model('weights/Stream_1_Frontalization_epoch-05_val-accu-0.29.hdf5')


'''
Customize the model
'''
# Add geometry input
input_geo = Input(shape=(136,), name='input-geometry')
# geo_layer = Flatten(name='flatten-geo')(input_geo)

# Obtain frontalization's last layer
last_layer = vgg_model_frontalization.get_layer('pool5').output
for layer in vgg_model_frontalization.layers:
   layer.name = layer.name + "-model_frontalization"
front_layer = Flatten(name='flatten-front')(last_layer)

# Concatenate geo and front layers and customize it
x = concatenate([front_layer, input_geo])
x = Dense(hidden_dim, activation='relu', name='fc6')(x)
x = Dense(hidden_dim, activation='relu', name='fc7')(x)
x = Dense(hidden_dim, activation='relu', name='fc8')(x)
out = Dense(nb_class, activation='softmax', name='fc9')(x)

custom_vgg_model = Model(
    [vgg_model_frontalization.input, input_geo], 
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

train_geo_datagen = ImageDataGenerator()
test_geo_datagen = ImageDataGenerator()

# Datagen
def obtain_datagen(datagen, train_path, seed_number=7, h5=False):
    return datagen.flow_from_directory(
                train_path,
                target_size=(img_height, img_width),
                batch_size=batch_size,
                seed=seed_number,
                class_mode='binary',
                h5=h5) 

# Training and Validation data generators
train_generator_1 = obtain_datagen(train_datagen, train_data_dir)
train_generator_2 = obtain_datagen(train_geo_datagen, train_data_dir_2, h5=True)

validation_generator_1 = obtain_datagen(test_datagen, validation_data_dir)
validation_generator_2 = obtain_datagen(test_geo_datagen, validation_data_dir_2, h5=True)

# Yield for data generators
def generate_data_generator_for_two_images(genX1, genX2):
    while True:
        # Get image
        X1i = genX1.next()
        
        # Get Geometry
        X2i = genX2.next()

        yield [X1i[0], X2i[0]], X1i[1]

# Yield for data generators
dataset_train_gen = generate_data_generator_for_two_images(train_generator_1, train_generator_2)
dataset_val_gen = generate_data_generator_for_two_images(validation_generator_1, validation_generator_2)


'''
Fine-tune the model
'''
# Freeze previous layers
for i, layer in enumerate(vgg_model_frontalization.layers):
    layer.trainable = False 

# Compile the model
custom_vgg_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

'''
Callbacks
'''
checkpointer = ModelCheckpoint(filepath='weights/'+model_name+'_epoch-{epoch:02d}_val-accu-{val_acc:.2f}.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('logs/'+model_name+'.log', separator=',', append=False)


'''
Train the model
'''
print('*************************************\nFine-tuning the model \n*************************************')
custom_vgg_model.fit_generator(
    dataset_train_gen,
    steps_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=dataset_val_gen,
    validation_steps=nb_validation_samples,
    callbacks=[checkpointer, csv_logger])