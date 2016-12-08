import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from helper import DataSet

path_to_dir = 'Data/cifar-10-batches-py'
batch_to_use = 'data_batch_'


def load_data(reshape=True):
    dicts = []
    for i in range(1, 6):
        with open(join(path_to_dir, batch_to_use + str(i)), 'rb') as fo:
            d = pickle.load(fo)
        # print(np.max((-127.5 + d['data'].reshape((-1, 32, 32, 3))) / 128.))
        dict = {'x': np.cast[np.float32]((-127.5 + d['data'].reshape((-1, 32, 32, 3), order='F'))/128.),
                'y': np.array(d['labels']).astype(np.uint8)}
        dicts.append(dict)

    data = np.concatenate([d['x'] for d in dicts],axis=0)
    labels = np.concatenate([d['y'] for d in dicts],axis=0)
    dataset = DataSet(data, labels, reshape=False)
    return dataset

def save_pic(data):
    f, axarr = plt.subplots(10, 10)
    images = data[:100, :, :, :]
    for i in range(100):
        axarr[int(i / 10), i % 10].axis('off')
        axarr[int(i / 10), i % 10].imshow(images[i])
    plt.savefig('cifar_actual')

if __name__ == '__main__':
    data = load_data()
    print('Max: %f, min: %f' % (np.max(data.images), np.min(data.images)))
    # save_pic(data.images)
