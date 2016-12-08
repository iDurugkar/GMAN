import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from helper import DataSet

path_to_dir = 'Data/celebA'
batch_to_use = 'celeb_64.npy'


def load_data(reshape=True):
    with open(join(path_to_dir, batch_to_use), 'rb') as fo:
        data = np.load(fo)
    data = data[:, 16:-16, 16:-16, :]
    data = np.cast[np.float32]((-127.5 + data)/128.)
    print(data.shape)
    # data = np.asarray(dict['data'])
    labels = np.zeros(data.shape[0])
    dataset = DataSet(data, labels, reshape=False)
    return dataset

def save_pic(data):
    f, axarr = plt.subplots(10, 10)
    images = data[:100, :, :, :]
    for i in range(100):
        axarr[int(i / 10), i % 10].axis('off')
        axarr[int(i / 10), i % 10].imshow(images[i], cmap='Greys_r')
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.savefig('celebA_actual')

if __name__ == '__main__':
    data = load_data()
    print('Max: %f, min: %f' % (np.max(data.images), np.min(data.images)))
    save_pic(data.images)
