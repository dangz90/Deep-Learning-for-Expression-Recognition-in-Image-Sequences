import os
import re
import h5py
import string
import numpy as np

def obtain_geometry_batch(batch_filenames, dataset):
    file_path = '2_Dataset/geometry/' + dataset + '/'

    ''' Loop through the files and assign label '''
    X = []
    Y = []
    labels = {'Anger_Fake': 0, 'Contentment_Real': 3, 'Sadness_Real': 9, 'Happiness_Fake': 6, 'Contentment_Fake': 2, 'Surprise_Fake': 10, 'Disgust_Fake': 4, 'Happiness_Real': 7, 'Anger_Real': 1, 'Disgust_Real': 5, 'Surprise_Real': 11, 'Sadness_Fake': 8}

    for filename in batch_filenames:
        # Get file path
        if dataset == 'training':
            fullpath = file_path + filename.replace('.jpg', '.h5').replace('_f','.MP4f')
        else:
            fullpath = file_path + filename.replace('.jpg', '.h5').replace('.mp4','_')

        # Extract from .h5 file
        f = h5py.File(fullpath, 'r')
        data = np.array(f.get('geometry'))

        Y.append(labels[filename.split('/')[0]])

        # Append to dataset
        try:
            data = data.reshape(1, 136, 1)
        except ValueError:
            print('\n')
            print('Error with', fullpath, data.shape)

        X.append(data)

    return np.array(X), Y