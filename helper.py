import tensorflow as tf
from tensorflow.python.framework import dtypes
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def get_mnist_data():
    mnist = input_data.read_data_sets('Data/MNIST', reshape=False)
    return mnist


def sigmoid(x):
    return 0.98 * tf.sigmoid(x) + 0.01


class DataSet(object):
    def __init__(self,
                 images,
                 labels,
                 fake_data=False,
                 one_hot=False,
                 dtype=dtypes.float32,
                 reshape=True):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
            self._num_examples = images.shape[0]

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            if reshape:
                assert images.shape[3] == 1
                images = images.reshape(images.shape[0],
                                        images.shape[1] * images.shape[2])

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)
                ]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def batch_norm(x, n_out, phase_train=tf.constant(True)):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def leaky_relu(alpha, x):
    return tf.maximum(alpha * x, x)


def maxout(x, n_input, n_output, n_maxouts=4, name='maxout'):
    with tf.name_scope(name):
        layer_output = None
        for i in range(n_maxouts):
            W = tf.Variable(tf.truncated_normal([n_input, n_output], mean=0.0, stddev=1. / n_output,
                                                dtype=tf.float32), name='W_%d' % i)
            b = tf.Variable(tf.zeros([n_output], dtype=tf.float32), name='b_%d' % i)
            # W = tf.get_variable('W_%d' % i, (n_input, n_output))
            # b = tf.get_variable('b_%d' % i, (n_output,))
            y = tf.matmul(x, W) + b
            if i == 0:
                layer_output = y
            else:
                layer_output = tf.maximum(layer_output, y)
    return layer_output


def weighted_arithmetic(w, x):
    numer = tf.reduce_sum(w*x)
    denom = tf.reduce_sum(w)
    return tf.div(numer, denom)


def dense(input, weight_shape, bias_shape):
    w = tf.get_variable('w', shape=weight_shape, initializer=tf.truncated_normal_initializer(stddev=0.02))
    b = tf.get_variable('b', shape=bias_shape, initializer=tf.constant_initializer(0.0))
    return tf.matmul(input, w) + b


def conv2d(input, filter_shape, bias_shape, stride=2, name='conv'):
    # with tf.variable_scope(name):
    filter = tf.get_variable('filter', shape=filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.02))
    bias = tf.get_variable('b', shape=bias_shape, initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, filter,
                        strides=[1, stride, stride, 1], padding='SAME')
    return conv + bias


def conv2d_alt(input, filter_shape, bias_shape, stride=2, name='conv'):
    with tf.variable_scope(name):
        filter = tf.get_variable('filter', shape=filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.02))
        bias = tf.get_variable('b', shape=bias_shape, initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input, filter,
                            strides=[1, stride, stride, 1], padding='SAME')
    return conv + bias


def mix_prediction(temp_ds, temp_losses, lam=0., mean_typ='arithmetic', weight_typ='normal', sign=-1.):
    # temp_ds & temp_losses are shape (# of discriminators)
    # output is a scalar

    if lam == 0.:
        weights = tf.ones_like(temp_losses)
    else:
        if weight_typ == 'log':
            weights = tf.pow(temp_losses, lam)
        else:
            weights = tf.exp(lam * temp_losses)

    if mean_typ == 'arithmetic':
        loss = weighted_arithmetic(weights, temp_losses)
    elif mean_typ == 'geometric':
        log_temp_losses = tf.log(sign*temp_losses)
        loss = sign*tf.exp(weighted_arithmetic(weights, log_temp_losses))
    elif mean_typ == 'harmonic':
        mn = tf.reduce_min(temp_losses) - 1e-3
        inv_temp_losses = tf.inv(temp_losses-mn)
        loss = mn + tf.inv(weighted_arithmetic(weights, inv_temp_losses))
    else:
        exponents = tf.exp(lam * temp_ds)
        loss = tf.div(tf.reduce_sum(tf.mul(temp_losses, exponents), reduction_indices=0),
                      tf.reduce_sum(exponents, reduction_indices=0))
    
    return loss
