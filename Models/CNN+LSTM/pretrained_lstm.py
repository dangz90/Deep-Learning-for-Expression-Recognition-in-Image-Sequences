import os
os.environ['KERAS_BACKEND']='tensorflow'
import keras
from keras.engine import Model
from keras_vggface.vggface import VGGFace
from keras.layers import Flatten, Dense, Input, Dropout

from utils import eucl_dist_ang, euclidean_distance


def define_model(n_output, lr, dropout, n_units, features_model):
    """
    Define model architecture, optimization and loss function
    """

    # Initialize weights
    weight_init = keras.initializers.glorot_uniform(seed=3)

    # Initialize optimizer
    opt = keras.optimizers.adam(lr)


    # Load pre-trained model
    if features_model is not None:
        cnn_model = keras.models.load_model(features_model, custom_objects={'euclidean_distance': euclidean_distance,
                                                                         'eucl_dist_ang': eucl_dist_ang})
    else:
        return 0

    model_input = keras.layers.Input(shape=(None,224,224,3), name='seq_input')

    # little hack to remove the last layers - we know that there are 7 layers after the last cnn layer, so we remove them
    # using pop
    cnn_model.layers.pop()
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

    cnn_model.summary()

    # x = keras.layers.TimeDistributed(keras.layers.Lambda(lambda x: cnn_model(x)))(model_input)
    x = keras.layers.TimeDistributed(Flatten())(model_input)

    x = keras.layers.LSTM(n_units, dropout=dropout, name='lstm')(x)
    out = Dense(n_output, kernel_initializer=weight_init, name='out')(x)

    model = Model(inputs=[model_input], outputs=out)

    model.summary()

    print(len(model.layers))
    print([n.name for n in model.layers])

    # Use mean euclidean distance as loss and angular error and mse as metric
    model.compile(loss=euclidean_distance,
                  optimizer=opt,
                  metrics=[eucl_dist_ang, 'mse'])

    return model
