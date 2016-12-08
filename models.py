# import tensorflow as tf
from helper import *
# from spatial_transformer import transformer


def generator(gan):
    with tf.name_scope('generator'):

        #  tf.placeholder(dtype=tf.float32, shape=[gan.batch_size, gan.num_latent])
        # _latent = tf.reshape(gan.latent, [None, 1, 1, gan.num_latent])
        with tf.name_scope('deconv0'):
            f = tf.Variable(tf.truncated_normal([3, 3, gan.num_hidden, gan.num_latent], mean=0.0,
                                                stddev=0.02, dtype=tf.float32),
                            name='filter')
            b = tf.Variable(tf.zeros([gan.num_hidden], dtype=tf.float32), name='b')
            h0 = tf.nn.bias_add(tf.nn.conv2d_transpose(gan.z, f,
                                                       [gan.channel_size, 4, 4, gan.num_hidden],
                                                       strides=[1, 4, 4, 1]), b)
            h0 = batch_norm(h0, gan.num_hidden)
            h0 = tf.nn.relu(h0)
        # with tf.name_scope('hidden1'):
        #     W = tf.Variable(tf.truncated_normal([gan.num_latent, gan.num_hidden*4*4], mean=0., stddev=0.02,
        #                                         dtype=tf.float32), name='W')
        #     b = tf.Variable(tf.zeros([gan.num_hidden*4*4], dtype=tf.float32), name='b')
        #     h0 = tf.nn.relu(tf.matmul(gan.latent, W) + b)
        #     h0 = tf.reshape(h0, shape=[gan.batch_size, 4, 4, gan.num_hidden], name='reshape')

        with tf.name_scope('deconv1'):
            f = tf.Variable(tf.truncated_normal([5, 5, gan.num_hidden / 2, gan.num_hidden], mean=0.0,
                                                stddev=0.02, dtype=tf.float32),
                            name='filter')
            b = tf.Variable(tf.zeros([gan.num_hidden / 2], dtype=tf.float32), name='b')
            h1 = tf.nn.bias_add(tf.nn.conv2d_transpose(h0, f,
                                                       [gan.channel_size, 8, 8, gan.num_hidden / 2],
                                                       strides=[1, 2, 2, 1]), b)
            h1 = batch_norm(h1, gan.num_hidden / 2)
            h1 = tf.nn.relu(h1)

        with tf.name_scope('deconv2'):
            f = tf.Variable(tf.truncated_normal([5, 5, gan.num_hidden / 4, gan.num_hidden / 2], mean=0.0,
                                                stddev=0.02, dtype=tf.float32),
                            name='filter')
            b = tf.Variable(tf.zeros([gan.num_hidden / 4], dtype=tf.float32), name='b')
            h2 = tf.nn.bias_add(tf.nn.conv2d_transpose(h1, f,
                                                       [gan.channel_size, 16, 16, gan.num_hidden / 4],
                                                       strides=[1, 2, 2, 1]), b)
            h2 = batch_norm(h2, gan.num_hidden / 4)
            h2 = tf.nn.relu(h2)

        if gan.side == 32:
            with tf.name_scope('gen_images'):
                f = tf.Variable(tf.truncated_normal([5, 5, gan.num_channels, gan.num_hidden / 4], mean=0.0,
                                                    stddev=0.02, dtype=tf.float32),
                                name='filter')
                b = tf.Variable(tf.zeros([gan.num_channels], dtype=tf.float32), name='b')
                gen_image = tf.nn.tanh(
                    tf.nn.bias_add(tf.nn.conv2d_transpose(h2, f,
                                                          [gan.channel_size, gan.side, gan.side,
                                                           gan.num_channels],
                                                          strides=[1, 2, 2, 1]), b))

        else:
            with tf.name_scope('deconv3'):
                f = tf.Variable(tf.truncated_normal([5, 5, gan.num_hidden / 8, gan.num_hidden / 4], mean=0.0,
                                                    stddev=0.02, dtype=tf.float32),
                                name='filter')
                b = tf.Variable(tf.zeros([gan.num_hidden / 8], dtype=tf.float32), name='b')
                h3 = tf.nn.bias_add(
                    tf.nn.conv2d_transpose(h2, f,
                                           [gan.channel_size, gan.side / 2, gan.side / 2, gan.num_hidden / 8],
                                           strides=[1, 2, 2, 1]), b)
                h3 = batch_norm(h3, gan.num_hidden / 8)
                h3 = tf.nn.relu(h3)

            with tf.name_scope('gen_images'):
                f = tf.Variable(tf.truncated_normal([5, 5, gan.num_channels, gan.num_hidden / 8], mean=0.0,
                                                    stddev=0.02, dtype=tf.float32),
                                name='filter')
                b = tf.Variable(tf.zeros([gan.num_channels], dtype=tf.float32), name='b')
                gen_image = tf.nn.tanh(
                    tf.nn.bias_add(tf.nn.conv2d_transpose(h3, f,
                                                          [gan.channel_size, gan.side, gan.side,
                                                           gan.num_channels],
                                                          strides=[1, 2, 2, 1]), b))
    return gen_image

# def spatial_transformation(gan, image):
#     with tf.name_scope('generator'):
#         with tf.name_scope('pt1'):
#             # latent_temp = tf.random_uniform(shape=[gan.channel_size, gan.num_latent], minval=-1., maxval=1.,
#             #                                 name='sz')
#             w = tf.Variable(tf.truncated_normal([gan.num_latent, gan.num_latent], mean=0., stddev=1e-2))
#             b = tf.Variable(tf.zeros([gan.num_latent]))
#             h1 = tf.nn.relu(tf.matmul(tf.squeeze(gan.latent), w) + b)
#         with tf.name_scope('pt2'):
#             # latent_temp = tf.random_uniform(shape=[gan.channel_size, gan.num_latent], minval=-1., maxval=1.,
#             #                                 name='sz')
#             w = tf.Variable(tf.truncated_normal([gan.num_latent, gan.num_latent], mean=0., stddev=1e-2))
#             b = tf.Variable(tf.zeros([gan.num_latent]))
#             h2 = tf.nn.relu(tf.matmul(h1, w) + b)
#         with tf.name_scope('pt3'):
#             # latent_temp = tf.random_uniform(shape=[gan.channel_size, gan.num_latent], minval=-1., maxval=1.,
#             #                                 name='sz')
#             w = tf.Variable(tf.truncated_normal([gan.num_latent, gan.num_latent], mean=0., stddev=1e-2))
#             b = tf.Variable(tf.zeros([gan.num_latent]))
#             h3 = tf.nn.relu(tf.matmul(h2, w) + b)
#         with tf.name_scope('transformation'):
#             # latent_temp = tf.random_uniform(shape=[gan.channel_size, gan.num_latent], minval=-1., maxval=1.,
#             #                                 name='sz')
#             w = tf.Variable(tf.truncated_normal([gan.num_latent, 6], mean=0., stddev=0.1), name='theta_weights')
#             initial = np.array([[1., 0, 0], [0, 1., 0]])
#             initial = initial.flatten()
#             b = tf.Variable(initial_value=initial, name='theta_b')
#             b = tf.cast(b, 'float32')
#             theta = tf.matmul(h3, w) + b
#             shape = image.get_shape().as_list()
#             t_image = transformer(image, theta, shape)
#     return t_image


def discriminator(gan, inp, num, keep_prob, reuse=False):
    gpu_num = 0
    if False:  # num % 2 == 1:
        hidden_units = gan.h_adv / 2  # / (2 ** int(num / 2))
        print(hidden_units)
        print(keep_prob)
        with tf.device('/gpu:%d' % gpu_num):
            with tf.name_scope('discriminator_%d' % num) and tf.variable_scope('discriminator_%d' % num):
                print('Got here')
                import sys
                sys.stdout.flush()
                if reuse:
                    tf.get_variable_scope().reuse_variables()
                with tf.name_scope('conv0') and tf.variable_scope('conv0'):
                    h0 = conv2d(inp, [3, 3, gan.num_channels, hidden_units / 2], [hidden_units / 2],
                                stride=1, name='h0')
                    h0 = leaky_relu(0.1, h0)
                    # if num % 3 == 1:
                    # h0 = conv2d(h0, [3, 3, hidden_units / 2, hidden_units / 2], [hidden_units / 2],
                    #             stride=1, name='h1')
                    # h0 = leaky_relu(0.1, h0)
                    # h0 = conv2d(h0, [3, 3, hidden_units / 2, hidden_units / 2], [hidden_units / 2],
                    #             stride=1, name='h12')
                    # h0 = leaky_relu(0.1, h0)
                    h0 = conv2d(h0, [3, 3, hidden_units / 2, hidden_units / 2], [hidden_units / 2], name='h3',
                                stride=2)
                    h0 = leaky_relu(0.1, h0)
                    h0 = tf.nn.dropout(h0, keep_prob)
                with tf.name_scope('conv1') and tf.variable_scope('conv1'):
                    h1 = conv2d(h0, [3, 3, hidden_units / 2, hidden_units], [hidden_units],
                                stride=1, name='h0')
                    h1 = leaky_relu(0.1, h1)
                    # if num % 2 == 1:
                    # h1 = conv2d(h1, [3, 3, hidden_units, hidden_units], [hidden_units],
                    #             stride=1, name='h1')
                    # h1 = leaky_relu(0.1, h1)
                    # h1 = conv2d(h1, [3, 3, hidden_units, hidden_units], [hidden_units],
                    #             stride=1, name='h12')
                    # h1 = leaky_relu(0.1, h1)
                    h1 = conv2d(h1, [3, 3, hidden_units, hidden_units], [hidden_units], name='h3', stride=2)
                    h1 = leaky_relu(0.1, h1)
                    h1 = tf.nn.dropout(h1, keep_prob)
                with tf.name_scope('conv2') and tf.variable_scope('conv2'):
                    h2 = conv2d(h1, [3, 3, hidden_units, hidden_units], [hidden_units],
                                stride=2, name='h0')
                    h2 = leaky_relu(0.1, h2)
                    # h2 = conv2d(h2, [3, 3, hidden_units, hidden_units], [hidden_units],
                    #             stride=1, name='h1')
                    # h2 = leaky_relu(0.1, h2)
                    # h2 = conv2d(h2, [3, 3, hidden_units, hidden_units], [hidden_units],
                    #             stride=1, name='h12')
                    # h2 = leaky_relu(0.1, h2)
                    # h2 = conv2d(h2, [3, 3, hidden_units, hidden_units], [hidden_units], name='h3', stride=2)
                    # h2 = leaky_relu(0.1, h2)
                    # h2 = tf.nn.dropout(h2, keep_prob)
                # if False:  # num % 2 == 0:
                #     with tf.name_scope('reshape') and tf.variable_scope('reshape'):
                #         shape = h1.get_shape().as_list()
                #         num_units = shape[1] * shape[2] * shape[3]
                #         flattened = tf.reshape(h1, [gan.batch_size, num_units])
                #         flattened = tf.nn.dropout(flattened, keep_prob=0.9)
                # else:
                #     with tf.name_scope('conv2') and tf.variable_scope('conv2'):
                #         h2 = conv2d(h1, [5, 5, hidden_units / 4, hidden_units / 2], [hidden_units / 2])
                #         h2 = leaky_relu(0.2, h2)
                #         h2 = tf.nn.dropout(h2, keep_prob)
                #
                #     if gan.side == 32:
                with tf.name_scope('reshape') and tf.variable_scope('reshape'):
                    shape = h2.get_shape().as_list()
                    print(shape)
                    print('That was shape of the convolution')
                    num_units = shape[1] * shape[2] * shape[3]
                    flattened = tf.reshape(h2, [gan.batch_size, num_units])
                    # flattened = tf.nn.dropout(flattened, keep_prob=0.9)
                    # else:
                    #     with tf.name_scope('conv3') and tf.variable_scope('conv3'):
                    #         h3 = conv2d(h2, [5, 5, hidden_units / 2, hidden_units], [hidden_units])
                    #         h3 = leaky_relu(0.2, h3)
                    #     with tf.name_scope('reshape') and tf.variable_scope('reshape'):
                    #         shape = h3.get_shape().as_list()
                    #         num_units = shape[1] * shape[2] * shape[3]
                    #         flattened = tf.reshape(h3, [gan.batch_size, num_units])
                    #         # flattened = tf.nn.dropout(flattened, keep_prob=0.9)
                with tf.name_scope('prediction') and tf.variable_scope('prediction'):
                    pred = dense(flattened, [num_units, 1], [1])
    else:
        hidden_units = gan.h_adv
        print(hidden_units)
        print(keep_prob)
        with tf.device('/gpu:%d' % gpu_num):
            with tf.name_scope('discriminator_%d' % num) and tf.variable_scope('discriminator_%d' % num):
                if reuse:
                    tf.get_variable_scope().reuse_variables()
                with tf.name_scope('conv0') and tf.variable_scope('conv0'):
                    h0 = conv2d(inp, [3, 3, gan.num_channels, hidden_units / 4], [hidden_units / 4],
                                stride=2, name='h0')
                    h0 = leaky_relu(0.2, h0)
                    h0 = tf.nn.dropout(h0, keep_prob)
                with tf.name_scope('conv1') and tf.variable_scope('conv1'):
                    h1 = conv2d(h0, [3, 3, hidden_units / 4, hidden_units / 2], [hidden_units / 2],
                                stride=2, name='h0')
                    h1 = leaky_relu(0.2, h1)
                    h1 = tf.nn.dropout(h1, keep_prob)
                # with tf.name_scope('conv2') and tf.variable_scope('conv2'):
                #     h2 = conv2d(h1, [3, 3, hidden_units / 2, hidden_units], [hidden_units],
                #                 stride=1, name='h0')
                #     h2 = leaky_relu(0.2, h2)
                #     # h2 = tf.nn.dropout(h2, keep_prob)
                if num % 2 == 1:
                    with tf.name_scope('reshape') and tf.variable_scope('reshape'):
                        shape = h1.get_shape().as_list()
                        num_units = shape[1] * shape[2] * shape[3]
                        flattened = tf.reshape(h1, [gan.batch_size, num_units])
                        flattened = tf.nn.dropout(flattened, keep_prob=0.9)
                else:
                    with tf.name_scope('conv2') and tf.variable_scope('conv2'):
                        h2 = conv2d(h1, [3, 3, hidden_units / 2, hidden_units], [hidden_units],
                                    stride=2, name='h0')
                        h2 = leaky_relu(0.2, h2)
                        h2 = tf.nn.dropout(h2, keep_prob)
                #
                #     if gan.side == 32:
                    with tf.name_scope('reshape') and tf.variable_scope('reshape'):
                        shape = h2.get_shape().as_list()
                        num_units = shape[1] * shape[2] * shape[3]
                        flattened = tf.reshape(h2, [gan.batch_size, num_units])
                        flattened = tf.nn.dropout(flattened, keep_prob=0.9)
                    # else:
                    #     with tf.name_scope('conv3') and tf.variable_scope('conv3'):
                    #         h3 = conv2d(h2, [5, 5, hidden_units / 2, hidden_units], [hidden_units])
                    #         h3 = leaky_relu(0.2, h3)
                    #     with tf.name_scope('reshape') and tf.variable_scope('reshape'):
                    #         shape = h3.get_shape().as_list()
                    #         num_units = shape[1] * shape[2] * shape[3]
                    #         flattened = tf.reshape(h3, [gan.batch_size, num_units])
                    #         # flattened = tf.nn.dropout(flattened, keep_prob=0.9)
                with tf.name_scope('prediction') and tf.variable_scope('prediction'):
                    pred = dense(flattened, [num_units, 1], [1])
    return pred
