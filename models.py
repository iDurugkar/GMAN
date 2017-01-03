import tensorflow as tf
from helper import batch_norm, leaky_relu, dense, conv2d


def generator(gan):
    with tf.variable_scope('generator'):

        with tf.variable_scope('deconv0'):
            f = tf.Variable(tf.truncated_normal([3, 3, gan.num_hidden, gan.num_latent], mean=0.0,
                                                stddev=0.02, dtype=tf.float32),
                            name='filter')
            b = tf.Variable(tf.zeros([gan.num_hidden], dtype=tf.float32), name='b')
            h0 = tf.nn.bias_add(tf.nn.conv2d_transpose(gan.z, f,
                                                       [gan.channel_size, 4, 4, gan.num_hidden],
                                                       strides=[1, 4, 4, 1]), b)
            h0 = batch_norm(h0, gan.num_hidden)
            h0 = tf.nn.relu(h0)

        with tf.variable_scope('deconv1'):
            f = tf.Variable(tf.truncated_normal([5, 5, gan.num_hidden / 2, gan.num_hidden], mean=0.0,
                                                stddev=0.02, dtype=tf.float32),
                            name='filter')
            b = tf.Variable(tf.zeros([gan.num_hidden / 2], dtype=tf.float32), name='b')
            h1 = tf.nn.bias_add(tf.nn.conv2d_transpose(h0, f,
                                                       [gan.channel_size, 8, 8, gan.num_hidden / 2],
                                                       strides=[1, 2, 2, 1]), b)
            h1 = batch_norm(h1, gan.num_hidden / 2)
            h1 = tf.nn.relu(h1)

        with tf.variable_scope('deconv2'):
            f = tf.Variable(tf.truncated_normal([5, 5, gan.num_hidden / 4, gan.num_hidden / 2], mean=0.0,
                                                stddev=0.02, dtype=tf.float32),
                            name='filter')
            b = tf.Variable(tf.zeros([gan.num_hidden / 4], dtype=tf.float32), name='b')
            h2 = tf.nn.bias_add(tf.nn.conv2d_transpose(h1, f,
                                                       [gan.channel_size, 16, 16, gan.num_hidden / 4],
                                                       strides=[1, 2, 2, 1]), b)
            h2 = batch_norm(h2, gan.num_hidden / 4)
            h2 = tf.nn.relu(h2)

        with tf.variable_scope('gen_images'):
            f = tf.Variable(tf.truncated_normal([5, 5, gan.num_channels, gan.num_hidden / 4], mean=0.0,
                                                stddev=0.02, dtype=tf.float32),
                            name='filter')
            b = tf.Variable(tf.zeros([gan.num_channels], dtype=tf.float32), name='b')
            gen_image = tf.nn.tanh(
                tf.nn.bias_add(tf.nn.conv2d_transpose(h2, f,
                                                      [gan.channel_size, gan.side, gan.side,
                                                       gan.num_channels],
                                                      strides=[1, 2, 2, 1]), b))
    return gen_image


def discriminator(gan, inp, num, keep_prob, reuse=False):
    gpu_num = 0
    hidden_units = gan.h_adv
    print(hidden_units)
    print(keep_prob)
    with tf.device('/gpu:%d' % gpu_num):
        with tf.variable_scope('discriminator_%d' % num):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            with tf.variable_scope('conv0'):
                h0 = conv2d(inp, [3, 3, gan.num_channels, hidden_units / 4], [hidden_units / 4],
                            stride=2, name='h0')
                h0 = leaky_relu(0.2, h0)
                h0 = tf.nn.dropout(h0, keep_prob)
            with tf.variable_scope('conv1'):
                h1 = conv2d(h0, [3, 3, hidden_units / 4, hidden_units / 2], [hidden_units / 2],
                            stride=2, name='h0')
                h1 = leaky_relu(0.2, h1)
                h1 = tf.nn.dropout(h1, keep_prob)
            with tf.variable_scope('conv2'):
                h2 = conv2d(h1, [3, 3, hidden_units / 2, hidden_units], [hidden_units],
                            stride=1, name='h0')
                h2 = leaky_relu(0.2, h2)
            with tf.variable_scope('reshape'):
                shape = h2.get_shape().as_list()
                num_units = shape[1] * shape[2] * shape[3]
                flattened = tf.reshape(h2, [gan.batch_size, num_units])
            with tf.variable_scope('prediction'):
                pred = dense(flattened, [num_units, 1], [1])
    return pred
