# export CUDA_VISIBLE_DEVICES=1,2
import os
import time
import six, csv
from vggface_model import VGGFace

from keras.models import  Model
from keras.optimizers import adam
from keras.layers import Flatten, Dense, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, Callback

from collections import OrderedDict
from SaveModelDirectory import create_path

__author__ = 'Daniel Garcia Zapata'


# Datagen
def obtain_datagen(datagen, path):
    return datagen.flow_from_directory(
                    path,
                    target_size=(img_height, img_width),
                    batch_size=batch_size,
                    class_mode='binary')

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

    print "Starting:", time.ctime()

    ###########################################
    # Parameters

    epochs = 20
    batch_size = 40

    ###########################################
    # Data

    img_width, img_height, channel = 224, 224, 3        # Resolution of inputs
    nb_class = 6

    dataset = 'OULU-CASIA'
    partition = 'preface'       # prealigned preface
    train_data_dir = os.path.join('..', '..', '_Dataset', dataset, partition, 'training')
    validation_data_dir = os.path.join('..', '..', '_Dataset', dataset, partition, 'validation')

    __model__ = 'CNN_' + partition    

    if dataset == 'OULU-CASIA':
        nb_train_samples = 5517
        nb_validation_samples = 449
    else:
        nb_train_samples = 29798
        nb_validation_samples = 4428
    
    nb_train_samples = nb_train_samples / batch_size
    nb_validation_samples = nb_validation_samples / batch_size    

    '''
    Load the model
    '''
    vgg_model = VGGFace(
    	include_top=False,
    	input_shape=(img_width, img_height, channel))

    '''
    Custom the model
    '''
    last_layer = vgg_model.get_layer('pool5').output
    x = Flatten(name='flatten')(last_layer)
    x = Dense(4096, activation='relu', name='fc6')(x)
    x = Dense(4096, activation='relu', name='fc7')(x)
    x = Dense(2622, activation='relu', name='fc8')(x)
    out = Dense(nb_class, activation='softmax', name='fc9')(x)

    model = Model(
    	vgg_model.input, 
    	out)

    '''
    Dataset Generators
    '''
    datagen = ImageDataGenerator(
        rescale=1. / 224,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_generator = obtain_datagen(datagen, train_data_dir)
    validation_generator = obtain_datagen(datagen, validation_data_dir)

    '''
    Freeze previous layers
    '''
    for i, layer in enumerate(vgg_model.layers):
    	layer.trainable = False 

    '''
    Compile the model
    '''
    lr = 0.0001
    optimizer = adam(lr=lr)
    loss = 'sparse_categorical_crossentropy'
    model.compile(  loss=loss,          # loss=euclidean_distance_loss,
                    optimizer=optimizer,        
                    metrics=['accuracy', 'top_k_categorical_accuracy'])

    '''
    Callbacks
    '''
    # Create Version folder
    export_path = create_path(__model__, dataset)

    checkpointer = ModelCheckpoint(filepath=os.path.join(export_path, __model__+'_epoch-{epoch:02d}_val-accu-{val_acc:.4f}.hdf5'), verbose=1) #, save_best_only=True) 
    csv_logger = CSVLogger(os.path.join(export_path, '_logs_'+__model__+'.log'), separator=',', append=False)
    tensorboard = TensorBoard(log_dir=export_path, histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    model_parameters = ModelParameters(os.path.join(export_path, '_model_'+__model__+'.log'))
    '''
    Train the model
    '''
    print('*************************************\nTraining \n*************************************')
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples,
        callbacks=[checkpointer, csv_logger, tensorboard, model_parameters])

    print("Ending:", time.ctime())
