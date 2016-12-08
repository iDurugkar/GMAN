import os
from dataset import Cifar, Celeb, GenericImages
import tensorflow as tf
import numpy as np
from helper import mix_prediction, sigmoid, get_mnist_data
import matplotlib.pyplot as plt
from models import generator, discriminator
from time import time


class GMAN:
    def __init__(self, num_latent, num_out, batch_size, num_disc, num_channels=3,
                 num_hidden=1024, D_weights=None, G_weights=None, name='GMAN',
                 mixing='arithmetic', weight_type='normal', objective='original',
                 boosting_variant=None):
        self.num_latent = num_latent
        self.side = num_out
        self.num_channels = num_channels
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.N = num_disc
        self.base_prob = 0.4
        self.delta_p = (0.5 - self.base_prob) / self.N
        self.h_adv = num_hidden
        self.name = name
        self.weight_type = weight_type
        self.channel_size = self.batch_size * self.N

        # boosting variables
        self.aux_vars = []
        self.aux_vars_new = []

        with tf.variable_scope(self.name):
            # Define latent distribution
            self.z = tf.random_uniform(shape=[self.channel_size, 1, 1, self.num_latent],
                                       minval=-1., maxval=1., name='z')
            
            # Generate fake images
            self.fake = generator(self)
            if boosting_variant is None:
                fake_split = tf.split(0, self.N, self.fake, name='fake_split')
            else:
                fake_split = [self.fake]*self.N

            # Discriminate fake images
            self.Df_logits = [discriminator(self, fake_split[ind], ind, (self.base_prob + self.delta_p * (ind + 1)),
                                            reuse=False)
                              for ind in range(self.N)]
            
            # Retrieve real images
            self.real = tf.placeholder(tf.float32, shape=[self.channel_size, self.side, self.side, self.num_channels],
                                       name='real')
            if boosting_variant is None:
                real_split = tf.split(0, self.N, self.real, name='real_split')
            else:
                real_split = [self.real]*self.N
            
            # Discriminate real images
            self.Dr_logits = [discriminator(self, real_split[ind], ind, (self.base_prob + self.delta_p * (ind + 1)),
                                            reuse=True)
                              for ind in range(self.N)]

            # Retrieve trainable weights
            t_vars = tf.trainable_variables()
            for var in t_vars:
                print(var.name)
            self.G_vars = [var for var in t_vars if (self.name + '/generator') in var.name]
            self.D_vars = [[var for var in t_vars if ('%s/discriminator_%d' % (self.name, num)) in var.name]
                           for num in range(self.N)]
            # print(self.G_vars)
            # print(self.D_vars)
            # import sys
            # sys.stdout.flush()
            
            # Assign values to weights (if given)
            self.assign_weights = []
            if D_weights is not None:
                for i in range(self.N):
                    for j in range(len(self.D_vars[i])):
                        self.assign_weights.append(tf.assign(self.D_vars[i][j], D_weights[i][j]))
            if G_weights is not None:
                for j in range(len(self.G_vars)):
                    self.assign_weights.append(tf.assign(self.G_vars[j], G_weights[j]))

            # Define Discriminator losses
            with tf.name_scope('D_Loss'):
                if boosting_variant is None:
                    self.get_D_losses(obj=objective)
                else:
                    self.get_D_boosted_losses(boosting_variant, obj=objective)

            # Define Generator losses
            with tf.name_scope('G_Loss'):
                if boosting_variant is None:
                    self.get_G_loss(mixing, obj=objective)
                else:
                    self.get_G_boosted_loss(boosting_variant, mixing, obj=objective)

            # Construct Discriminator updates
            self.lrd = tf.placeholder(dtype=tf.float32)
            self.D_optimizer = tf.train.AdamOptimizer(learning_rate=self.lrd, beta1=0.5)
            self.D_optim = [self.D_optimizer.minimize(D_loss, var_list=self.D_vars[ind])
                            for ind, D_loss in enumerate(self.D_losses)]

            # Construct Generator updates
            self.lrg = tf.placeholder(dtype=tf.float32)
            self.G_optimizer = tf.train.AdamOptimizer(learning_rate=self.lrg, beta1=0.5)
            self.G_optim = self.G_optimizer.minimize(self.G_loss, var_list=self.G_vars)
            self.G_grads = self.G_optimizer.compute_gradients(self.G_loss, var_list=self.G_vars)
            
            # Join all updates
            self.all_opt = [opt for opt in self.D_optim]
            self.all_opt.append(self.G_optim)

    def get_D_boosted_losses(self, boosting_variant, obj='original'):
        # Define auxiliary placeholds
        t = tf.placeholder(tf.float32)
        alpha = tf.placeholder(tf.float32, shape=[self.N])
        v = tf.placeholder(tf.float32, shape=[self.N])

        # Compute expectation of booster prediction
        _Df_logits = tf.concat(concat_dim=1, values=self.Df_logits)
        _Dr_logits = tf.concat(concat_dim=1, values=self.Dr_logits)
        _Df = tf.cumsum(alpha*_Df_logits,axis=1,exclusive=False)
        _Dr = tf.cumsum(alpha*_Dr_logits,axis=1,exclusive=False)
        Df_weighted = v/tf.reduce_sum(v)*_Df
        Dr_weighted = v/tf.reduce_sum(v)*_Dr
        self.Df_expected = tf.reduce_sum(Df_weighted,reduction_indices=1)
        self.Dr_expected = tf.reduce_sum(Dr_weighted,reduction_indices=1)

        # Compute auxiliary variable, s
        # Note: 'q' is 'z' from AdaBoost.OL to avoid confusion with latent variable 'z' in GAN
        qf = -_Df_logits
        qr = _Dr_logits
        q = tf.concat(concat_dim=0, values=[qf,qr])
        s_0 = tf.clip_by_value(tf.cumsum(alpha*q, exclusive=True), -4., 4.)
        s_1 = tf.clip_by_value(tf.cumsum(alpha*q, exclusive=False), -4., 4.)

        # Compute loss weights
        w = 1/(1+tf.exp(s_0))  # size: batch_size x num_discriminators
        wf, wr = tf.split(split_dim=0, num_split=2, value=w)
        wf_split = tf.split(split_dim=1, num_split=self.N, value=wf)
        wr_split = tf.split(split_dim=1, num_split=self.N, value=wr)

        # Define v update -- only needed if training generator with expectation of booster prediction
        wrong_f = sigmoid(Df_weighted)
        wrong_r = sigmoid(-Dr_weighted)
        wrong = tf.concat(concat_dim=0, values=[wrong_f,wrong_r])
        v_new = tf.reduce_mean(v*tf.exp(wrong),reduction_indices=0)

        # Define alpha update
        nt = 4/tf.sqrt(t)
        alpha_delta = nt * q / (1 + tf.exp(s_1))
        alpha_new = tf.reduce_mean(tf.clip_by_value(alpha + alpha_delta, -2, 2), reduction_indices=0)

        # Store auxiliary variable update pairs (t,alpha,v)
        self.aux_vars = [t, alpha, v]
        self.aux_vars_new = [t+1, alpha_new, v_new]

        # logits --> probabilities
        self.Df = [sigmoid(logit) for logit in self.Df_logits]
        self.Dr = [sigmoid(logit) for logit in self.Dr_logits]
        self.min_Df = tf.reduce_min(self.Df)
        self.max_Df = tf.reduce_max(self.Df)
        self.min_Dr = tf.reduce_min(self.Dr)
        self.max_Dr = tf.reduce_max(self.Dr)
        tf.scalar_summary('D_0_z', tf.reduce_mean(self.Df[0]))
        tf.scalar_summary('min_D_z', self.min_Df)
        tf.scalar_summary('max_D_z', self.max_Df)
        tf.scalar_summary('D_0_x', tf.reduce_mean(self.Dr[0]))
        tf.scalar_summary('min_D_x', self.min_Dr)
        tf.scalar_summary('max_D_x', self.max_Dr)

        # Define discriminator losses
        if obj == 'original':
            self.D_losses = [tf.reduce_mean(-wr_split[ind]*tf.log(self.Dr[ind])
                                            - wf_split[ind]*tf.log(1-self.Df[ind]))
                                   for ind in range(len(self.Dr))]
        else:
            self.D_losses = [tf.reduce_mean(-wr_split[ind]*tf.log(self.Dr[ind])
                                            + wf_split[ind]*tf.log(self.Df[ind]))
                                   for ind in range(len(self.Dr))]
        for ind in range(len(self.Dr)):
            tf.scalar_summary('D_%d_Loss' % ind, self.D_losses[ind])

        # Define minimax objectives for discriminators
        self.V_D = [tf.reduce_mean(tf.log(self.Dr[ind])+tf.log(1-self.Df[ind])) for ind in range(len(self.Dr))]

    def get_G_boosted_loss(self, boosting_variant, mixing,obj='original'):
        # Define lambda placeholder
        self.l = tf.placeholder(tf.float32, name='lambda')

        # Boosting variants
        # boost_prediction: Use booster to predict probabilities
        # boost_training: Use boosting to train, but not predict probabilites
        if boosting_variant == 'boost_prediction':
            # Define generator loss
            if obj == 'original':
                self.G_loss = tf.reduce_mean(tf.log(1-sigmoid(self.Df_expected)))
            else:
                self.G_loss = tf.reduce_mean(-tf.log(sigmoid(self.Df_expected)))

            # Define minimax objective for generator
            self.V_G = tf.reduce_mean(tf.log(self.Dr_expected)+tf.log(1-sigmoid(self.Df_expected)))
        else:
            # Define generator loss
            if obj == 'original':
                self.G_losses = [tf.reduce_mean(tf.log(1-self.Df[ind]))
                                 for ind in range(len(self.Df))]
                sign = -1.
            else:
                self.G_losses = [tf.reduce_mean(-tf.log(self.Df[ind]))
                                 for ind in range(len(self.Df))]
                sign = 1.
            _G_losses = [tf.expand_dims(loss, 0) for loss in self.G_losses]
            _G_losses = tf.concat(0,_G_losses)
            self.G_loss = mix_prediction(_G_losses, self.l,
                                         mean_typ=mixing, weight_typ=self.weight_type,
                                         sign=sign)

            # Define minimax objective for generator
            self.V_G = mix_prediction(self.V_D, self.l,
                                      mean_typ=mixing, weight_typ=self.weight_type,
                                      sign=sign)

        tf.scalar_summary('G_loss', self.G_loss)

    def get_D_losses(self, obj='original'):
        # logits --> probabilities
        self.Df = [sigmoid(logit) for logit in self.Df_logits]
        self.Dr = [sigmoid(logit) for logit in self.Dr_logits]
        self.min_Df = tf.reduce_min(self.Df)
        self.max_Df = tf.reduce_max(self.Df)
        self.min_Dr = tf.reduce_min(self.Dr)
        self.max_Dr = tf.reduce_max(self.Dr)
        tf.scalar_summary('D_0_z', tf.reduce_mean(self.Df[0]))
        tf.scalar_summary('min_D_z', self.min_Df)
        tf.scalar_summary('max_D_z', self.max_Df)
        tf.scalar_summary('D_0_x', tf.reduce_mean(self.Dr[0]))
        tf.scalar_summary('min_D_x', self.min_Dr)
        tf.scalar_summary('max_D_x', self.max_Dr)

        # Define discriminator losses
        # if obj == 'original':
        self.D_losses = [tf.reduce_mean(-tf.log(self.Dr[ind])-tf.log(1-self.Df[ind]))
                         for ind in range(len(self.Dr))]
        # else:
        #     self.D_losses = [tf.reduce_mean(-tf.log(self.Dr[ind])-tf.log(1-self.Df[ind]))
        #                      for ind in range(len(self.Dr))]
        for ind in range(len(self.Dr)):
            tf.scalar_summary('D_%d_Loss' % ind, self.D_losses[ind])

        # Define minimax objectives for discriminators
        self.V_D = [tf.reduce_mean(tf.log(self.Dr[ind]) + tf.log(1-self.Df[ind])) for ind in range(len(self.Dr))]

    def get_G_loss(self, mixing, obj='original'):
        # Define lambda placeholder
        self.l = tf.placeholder(tf.float32, name='lambda')

        # Define generator loss
        if obj == 'original':
            self.G_losses = [tf.reduce_mean(tf.log(1-self.Df[ind]))
                             for ind in range(len(self.Df))]
            sign = -1.
        else:
            self.G_losses = [tf.reduce_mean(-tf.log(self.Df[ind]))
                             for ind in range(len(self.Df))]
            sign = 1.
        _G_losses = [tf.expand_dims(loss, 0) for loss in self.G_losses]
        _G_losses = tf.concat(0, _G_losses)
        self.G_loss = mix_prediction(_G_losses, self.l,
                                     mean_typ=mixing, weight_typ=self.weight_type,
                                     sign=sign)
        tf.scalar_summary('G_loss', self.G_loss)

        # Define minimax objectives for generator
        self.V_G = mix_prediction(self.V_D, self.l,
                                  mean_typ=mixing, weight_typ=self.weight_type,
                                  sign=sign)



def main(_):
    if FLAGS.path is None:
        path = FLAGS.dataset + '/%s_%d_%s_%s_%d' % (FLAGS.dataset, FLAGS.num_disc, FLAGS.mixing, FLAGS.weighting,
                                                    int(time()))
    else:
        path = FLAGS.path
    if not os.path.isdir(path):
        os.makedirs(path)

    if FLAGS.dataset == 'mnist':
        data = get_mnist_data().train
        data._images = np.pad((data._images - 127.5) / 128., ((0, 0), (2, 2), (2, 2), (0, 0)), 'minimum')
        print(data.images.shape)
        num_c = 1
    elif FLAGS.dataset == 'celebA':
        celeb = Celeb()
        data = celeb.load_data()
        num_c = 3
    elif FLAGS.dataset == 'cifar':
        cifar = Cifar()
        data = cifar.load_data()
        num_c = 3
    else:
        gen_data = GenericImages()
        data = gen_data.load_data(FLAGS.dataset)
        num_c = data.images.shape[-1]
        print(data.images.shape)
    print('Max: %f, Min: %f' % (np.max(data.images), np.min(data.images)))

    # Retrieve parameters
    lr = FLAGS.learning_rate
    batch_size = FLAGS.batch_size
    num_epochs = FLAGS.epochs
    iterations = FLAGS.iterations
    lam = FLAGS.lam
    boosting_variant = FLAGS.boosting

    # Plot random images from dataset
    c = 'Greys_r'
    indices = np.random.permutation(range(data.images.shape[0]))[:FLAGS.batch_size]
    images = data.images[indices]
    plot_fakes(images,num_c,batch_size,c,filename=path+'/real_baseline')

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        with tf.device('/gpu:0'):
            # Construct GMAN and run initializer
            gman = GMAN(FLAGS.latent, FLAGS.image_size, FLAGS.batch_size, FLAGS.num_disc,
                        num_channels=num_c, num_hidden=FLAGS.num_hidden,
                        mixing=FLAGS.mixing, weight_type=FLAGS.weighting,
                        objective=FLAGS.objective, boosting_variant=boosting_variant)

            # Initialize feed_dict
            feed_dict = {gman.real: None, gman.l: lam, gman.lrg: lr, gman.lrd: lr}

            # Initialize auxiliary variables
            if boosting_variant:
                t = 1
                alpha = np.zeros(gman.N, dtype=np.float32)
                v = np.ones(gman.N, dtype=np.float32)
                aux_vars_init = zip(gman.aux_vars, [t, alpha, v])
                feed_dict.update(aux_vars_init)

            train_writer = tf.train.SummaryWriter(path+'/',sess.graph)
            sum = tf.merge_all_summaries()
            init = tf.global_variables_initializer()
            sess.run(init)

            try:
                for j in range(num_epochs):
                    print('Epoch: %d' % j)
                    print('lambda: %0.2f' % lam)

                    D_losses = []
                    G_loss = []
                    V = []
                    min_Dr = []
                    max_Dr = []
                    min_Df = []
                    max_Df = []

                    for k in range(iterations):
                        feed_dict[gman.real] = data.next_batch(batch_size * gman.N)[0]

                        summary, _G_loss, _V, _min_Df, _max_Df, _min_Dr, _max_Dr = sess.run([sum, gman.G_loss, gman.V_G, gman.min_Df, gman.max_Df, gman.min_Dr, gman.max_Dr],
                                                                 feed_dict=feed_dict)
                        train_writer.add_summary(summary, j * iterations + k)

                        _D_losses = sess.run(gman.D_losses, feed_dict=feed_dict)

                        aux_vars_new = sess.run([gman.all_opt]+gman.aux_vars_new, feed_dict=feed_dict)
                        if len(aux_vars_new) > 1:
                            aux_vars_new = aux_vars_new[1:]
                            feed_dict.update(zip(gman.aux_vars, aux_vars_new))

                        G_loss.append(_G_loss)
                        D_losses.append(_D_losses)
                        V.append(_V)
                        min_Df.append(_min_Df)
                        max_Df.append(_max_Df)
                        min_Dr.append(_min_Dr)
                        max_Dr.append(_max_Dr)

                        if (k + 1) % 10 == 0:
                            print('epoch %d, minibatch: %d, D_Loss: %0.4f, G_Loss: %0.4f, V: %0.4f, Df: [%0.4f,%0.4f], Dr: [%0.4f,%0.4f]'
                                  % (j, k, np.mean(D_losses), np.mean(G_loss), np.mean(V), min(min_Df), max(max_Df), min(min_Dr), max(max_Dr)))
                            G_loss = []
                            D_losses = []
                            V = []
                            min_Df = []
                            max_Df = []
                            min_Dr = []
                            max_Dr = []

                    images = sess.run(gman.fake, feed_dict=feed_dict)[:batch_size]
                    plot_fakes(images,  num_c, batch_size, c, filename=path+'/%d.png' % j)

                    with tf.device('/cpu:0'):
                        saver = tf.train.Saver()
                        mpath = saver.save(sess, path + '/model.ckpt')
                        print('Model saved as %s' % mpath)

            except KeyboardInterrupt:
                print('interrupted run')
                with tf.device('/cpu:0'):
                    saver = tf.train.Saver()
                    mpath = saver.save(sess, path + '/model.ckpt')
                    print('Model saved as %s' % mpath)

            images = sess.run(gman.fake, feed_dict=feed_dict)[:batch_size]
            plot_fakes(images,num_c,batch_size,c,filename=path+'/final.png')
            

def plot_fakes(images,num_c,batch_size,c,filename):
    f, axarr = plt.subplots(10, 10)
    images = (np.add(images, 1.) / 2.)
    if num_c == 1:
        images = np.squeeze(images)
    for i in range(batch_size):
        axarr[int(i / 10), i % 10].axis('off')
        axarr[int(i / 10), i % 10].imshow(images[i], cmap=c)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_integer("epochs", 25, "Epoch to train [25]")
    flags.DEFINE_integer("iterations", 600, "Iterations per epoch [600]")
    flags.DEFINE_integer("latent", 100, "number of latent variables. [100]")
    flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
    # flags.DEFINE_float("dropout", 0.7, "dropout probability. [0.7]")
    flags.DEFINE_float("lam", 1., "Factor controlling how much the mixing moves towards a max. [1.]")
    flags.DEFINE_integer("batch_size", 100, "The size of batch images [100]")
    flags.DEFINE_integer("image_size", 32, "The size of the output images to produce [64]")
    # flags.DEFINE_integer("num_c", 1, "Number of channels. 3 for RGB [3]")
    flags.DEFINE_integer("num_disc", 5, "Number of discriminators. [5]")
    flags.DEFINE_integer("num_hidden", 512, "Number of hidden units. [512]")
    flags.DEFINE_string("dataset", "mnist", "The name of dataset [celebA, mnist, cifar]")
    flags.DEFINE_string("path", None, "The name of path to save at")
    flags.DEFINE_string("mixing", "arithmetic", "Mixing type [arithmetic, geometric, harmonic]")
    flags.DEFINE_string("boosting", None, "Mixing type [boost_prediction, boost_training, None]")
    flags.DEFINE_string("weighting", "normal", "Mixing type [normal, log]")
    flags.DEFINE_string("objective", "original", "Generator objective [original, modified]")
    # flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
    # flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
    FLAGS = flags.FLAGS

    tf.app.run()
