import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from helper import DataSet
from PIL import Image


def load_data(path_to_data, reshape=True):
    # Check if data is a numpy array
    if path_to_data.endswith('.npy'):
        with open(path_to_data, 'rb') as fo:
            data = np.load(fo).astype(np.float32)
    # Assuming data is image files in a directory
    else:
        from scipy import misc
        files_read = []
        for root, subFolders, files in os.walk(path_to_data):
            print(root)
            print(subFolders)
            print(len(files))
            for f in files:
                if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg'):
                    files_read.append(os.path.join(root, f))
                    # print(files_read[-1])
            print('one subdir done')
        # files = [f for f in os.listdir(path_to_data) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]
        print('Done listing files')
        images = []
        for f in files_read:
            try:
                # im = misc.imread(f)
                im = Image.open(f)
                im = np.array(im)
                # print(im)
            except IOError:
                print('Could not read: %s' % f)
            if len(im.shape) == 2:
                im = np.expand_dims(im, -1)
            images.append(im)
        print('Done reading files')
        num_c = images[0].shape[-1]
        for i in range(len(images)):
            images[i] = misc.imresize(images[i], (32, 32, num_c))
            # if len(images[i].shape) == 3:
            #     images[i] = np.expand_dims(images[i], 0)
        data = np.stack(images, axis=0).astype(np.float32)

    # print(np.max((-127.5 + d['data'].reshape((-1, 32, 32, 3))) / 128.))
    data /= np.float32((data.max() - data.min() + 1.) / 2.)
    data = data.astype(np.float32)
    data -= np.float32(1.)
    print('Dataset shape is:')
    print(data.shape)
    labels = np.zeros(shape=data.shape[0])
    dataset = DataSet(data, labels, reshape=False)
    return dataset

def save_pic(data):
    f, axarr = plt.subplots(10, 10)
    images = data[:100, :, :, :]
    for i in range(100):
        axarr[int(i / 10), i % 10].axis('off')
        axarr[int(i / 10), i % 10].imshow(images[i])
    plt.savefig('data_actual')

if __name__ == '__main__':
    data = load_data()
    print('Max: %f, min: %f' % (np.max(data.images), np.min(data.images)))
    # save_pic(data.images)
