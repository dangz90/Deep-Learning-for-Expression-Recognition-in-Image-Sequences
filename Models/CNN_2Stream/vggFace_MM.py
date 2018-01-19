# export CUDA_VISIBLE_DEVICES=1,2
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense, Input, concatenate
from keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.metrics import confusion_matrix

'''
Model Parameters
'''
# dimensions of our images.
model_name = 'Fuse_1Frontalization_2Patch_Freezed_MaxPooling'
img_width, img_height = 224, 224 # Resolution of inputs
channel = 3
train_samples = 29799
validation_samples = 4428
epochs = 20
batch_size = 40
nb_train_samples = 29799 / batch_size
nb_validation_samples = 4428 / batch_size

# Dataset for Stream 1
train_data_dir_1 = '2_Dataset/frontalization/training/NoSymmetry/'
validation_data_dir_1 = '2_Dataset/frontalization/validation/NoSymmetry/'
# Dataset for Stream 2
train_data_dir_2 = '2_Dataset/patches/training/'
validation_data_dir_2 = '2_Dataset/patches/validation/'

# Custom parameters
nb_class = 12
hidden_dim = 512


'''
Load the models
'''
vgg_model_frontalization = load_model('weights/Stream_1_Frontalization.epoch-00_val-accu-0.24.hdf5')
vgg_model_patch = load_model('weights/Stream_2_Patches.epoch-00_val-accu-0.30.hdf5')

'''
Customize the model
'''
stream_1 = vgg_model_frontalization.get_layer('pool5').output
stream_1 = Flatten(name='flatten-1')(stream_1)

for layer in vgg_model_frontalization.layers:
   layer.name = layer.name + "-model_frontalization"

stream_2 = vgg_model_patch.get_layer('pool5').output
stream_2 = Flatten(name='flatten-2')(stream_2)

for layer in vgg_model_frontalization.layers:
   layer.name = layer.name + "-model_patches"

# fuse_layer = stream_1 + stream_2
fuse_layer = concatenate([stream_1, stream_2])

fuse_layer = Dense(hidden_dim, activation='relu', name='fc6')(fuse_layer)
fuse_layer = Dense(hidden_dim, activation='relu', name='fc7')(fuse_layer)
fuse_layer = Dense(hidden_dim, activation='relu', name='fc8')(fuse_layer)
out = Dense(nb_class, activation='softmax', name='fc9')(fuse_layer)

custom_vgg_model = Model(
	[vgg_model_frontalization.input, vgg_model_patch.input],
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

# Datagen
def obtain_datagen(datagen, train_path, seed_number=7):
    return datagen.flow_from_directory(
                train_path,
                target_size=(img_height, img_width),
                batch_size=batch_size,
                seed=seed_number,
                class_mode='binary')

# Training data generators
train_generator_1 = obtain_datagen(train_datagen, train_data_dir_1)
train_generator_2 = obtain_datagen(train_datagen, train_data_dir_2)

# Validation data generators
validation_generator_1 = obtain_datagen(test_datagen, validation_data_dir_1)
validation_generator_2 = obtain_datagen(test_datagen, validation_data_dir_2)

# Yield for data generators
def generate_data_generator_for_two_images(genX1, genX2):
    while True:
            X1i = genX1.next()
            X2i = genX2 .next()
            yield [ X1i[0], X2i[0] ], X1i[1]

# Yield for data generators
dataset_train_gen = generate_data_generator_for_two_images(train_generator_1, train_generator_2)
dataset_val_gen = generate_data_generator_for_two_images(validation_generator_1, validation_generator_2)


'''
Fine-tune the model
'''
# Freeze previous layers
for i, layer in enumerate(vgg_model_frontalization.layers):
	layer.trainable = False
for i, layer in enumerate(vgg_model_patch.layers):
	layer.trainable = False 

# Compile the model
custom_vgg_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


'''
Callbacks
'''
checkpointer = ModelCheckpoint(filepath='weights/'+model_name+'.epoch-{epoch:02d}_val-accu-{val_acc:.2f}.hdf5', verbose=1, save_best_only=True)
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