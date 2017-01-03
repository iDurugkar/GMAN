# import tensorflow as tf
# import numpy as np
from helper import *
from dataset import Celeb, Cifar, GenericImages
import os
from GMAN import GMAN


def initialize_net_in_different_graph(path, num_latent, image_size, batch_size,
                                      num_disc, num_c, num_hidden=256,
                                      mixing='arithmetic', weighting='normal',
                                      objective='original', boosting_variant=None,
                                      self_learnt=False, name='GMAN', config=None):
    graph = tf.Graph()
    with graph.as_default():
        gan = GMAN(num_latent, image_size, batch_size, num_disc,
                   num_channels=num_c, num_hidden=num_hidden,
                   mixing=mixing, weight_type=weighting,
                   objective=objective, boosting_variant=boosting_variant,
                   self_challenged=self_learnt, name=name)
        saver = tf.train.Saver()
        sess = tf.Session(graph=graph, config=config)
        saver.restore(sess, path)
    # tf.placeholder()
    return graph, sess, gan


def main(_):
    num_latent = FLAGS.latent
    image_size = 32
    num_hidden = 256
    num_iterations = 5
    batch_size = FLAGS.batch_size
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    if FLAGS.dataset == 'mnist':
        data = get_mnist_data().train
        data._images = np.pad((data._images - 127.5) / 128., ((0, 0), (2, 2), (2, 2), (0, 0)), 'minimum')
        print(data.images.shape)
        num_c = 1
    elif FLAGS.dataset == 'celebA':
        celeb = Celeb()
        data = celeb.load_data()
        num_c = 3
    else:
        cifar = Cifar()
        data = cifar.load_data()
        num_c = 3
    print('Max: %f, Min: %f' % (np.max(data.images), np.min(data.images)))
    order = ['modified', 'original', '5_0', '5_1', '2_0', '2_1']  # 'self', 'arm_0', 'arm_1']  # , 'harm_0', 'harm_1']
    #  , 'boost' 'original', 'max',
    # order = ['baseline', 'arm_0', 'arm_1', 'harm_0']
    comparisons = {}
    for ind1 in range(len(order)):
        for ind2 in range( len(order)):
            name = order[ind1] + '_' + order[ind2]
            comparisons[name] = []

    # Assuming we are comparing 2 and 5 discriminators
    # num_disc = 2
    # iteration = 4

    # path, num_latent, image_size, batch_size,
    # num_disc, num_c, num_hidden = 256,
    # mixing = 'arithmetic', weighting = 'normal',
    # objective = 'original', boosting_variant = None,
    # self_learnt = False, name = 'GMAN', config = None
    for iteration in range(1, 2):
        models = {}
        for model in order:
            if model == 'modified':
                path = '%s/1_modified_%d_%d/model.ckpt' % (FLAGS.path, num_hidden, iteration)
                print(path)
                num_disc = 1
                lam = 1.0
                objective = 'modified'
                name = model
                self_learnt = False
            elif model == 'original':
                path = '%s/1_original_%d_%d/model.ckpt' % (FLAGS.path, num_hidden, iteration)
                print(path)
                num_disc = 1
                lam = 1.0
                objective = 'original'
                name = model
                self_learnt = False
            else:
                num_disc, lam = model.split('_')
                num_disc = int(num_disc)
                if lam is not 'self':
                    lam = float(lam)
                    self_learnt = False
                else:
                    self_learnt = True
                path = '%s/%s_%d_%d/model.ckpt' % (FLAGS.path, model, num_hidden, iteration)
                print(path)
                num_disc = 1
                objective = 'original'
                name = model
            models[model] = initialize_net_in_different_graph(path, num_latent, image_size, batch_size, num_disc,
                                                              num_c, num_hidden=num_hidden, objective=objective,
                                                              self_learnt=self_learnt, name=name)

        original_scores = {}
        images = [data.next_batch(batch_size * 5)[0] for _ in range(num_iterations)]
        for model in order:
            _, sess, gan = models[model]
            lam = 0.
            if model.endswith('1'):
                lam = 1.
            if model == 'original' or model == 'modified' or model == 'boost':
                num_im = batch_size
            else:
                num_disc = int(model.split('_')[0])
                num_im = batch_size * num_disc
            score = [sess.run(gan.eval_loss, feed_dict={gan.reals: images[i][:num_im], gan.training: True, gan.l: lam})
                     for i in range(num_iterations)]
            original_scores[model] = score
            print('calculated score for %s' % model)
        print('Calculated the original scores')

        # Exchange weights and run
        for ind1 in range(len(order)):
            m1 = order[ind1]
            graph1, sess1, gan1 = models[m1]
            gen1_weights = sess1.run(gan1.gen_vars)
            lam1 = 0.
            if m1.endswith('1'):
                lam1 = 1.
            if m1 == 'original' or m1 == 'modified' or m1 == 'boost':
                num_im1 = batch_size
            else:
                num_disc = int(m1.split('_')[0])
                num_im1 = batch_size * num_disc
            if m1 == 'self':
                gen1_weights = gen1_weights[:-1]
            print(len(gen1_weights))
            for ind2 in range(len(order)):
                m2 = order[ind2]
                lam2 = 0.
                if m2.endswith('1'):
                    lam2 = 1.
                if m2 == 'original' or m2 == 'modified' or m2 == 'boost':
                    num_im2 = batch_size
                else:
                    num_disc = int(m1.split('_')[0])
                    num_im2 = batch_size * num_disc
                graph2, sess2, gan2 = models[m2]
                gen2_weights = sess2.run(gan2.gen_vars)
                if m2 == 'self':
                    gen2_weights = gen2_weights[:-1]
                print(len(gen2_weights))
                with graph1.as_default():
                    assign_weights = []
                    for j in range(len(gen1_weights)):
                        assign_weights.append(tf.assign(gan1.gen_vars[j], gen2_weights[j]))
                    for assignment in assign_weights:
                        sess1.run(assignment)
                print('Swapped out weights for generator 1')
                with graph2.as_default():
                    assign_weights = []
                    for j in range(len(gen2_weights)):
                        assign_weights.append(tf.assign(gan2.gen_vars[j], gen1_weights[j]))
                    for assignment in assign_weights:
                        sess2.run(assignment)
                print('Swapped out weights for generator 2')
                for i in range(num_iterations):
                    s12 = sess1.run(gan1.eval_loss, feed_dict={gan1.reals: images[i][:num_im1], gan1.training: True, gan1.l: lam1})
                    ratio1 = s12 / original_scores[order[ind1]][i]
                    s21 = sess2.run(gan2.eval_loss, feed_dict={gan2.reals: images[i][:num_im2], gan2.training: True, gan2.l: lam2})
                    ratio2 = s21 / original_scores[order[ind2]][i]
                    # score = ratio1 / ratio2
                    # score = np.log(s12) - np.log(original_scores[order[ind1]][i]) - np.log(s21) + np.log(original_scores[order[ind2]][i])
                    score = np.log(ratio1) - np.log(ratio2)
                    comparisons[m1 + '_' + m2].append(score)
                print('Comparison done for: %s' % m1 + '_' + m2)
                with graph2.as_default():
                    assign_weights = []
                    for j in range(len(gen2_weights)):
                        assign_weights.append(tf.assign(gan2.gen_vars[j], gen2_weights[j]))
                    for assignment in assign_weights:
                        sess2.run(assignment)
                print('Swapped out weights for generator 2')
            with graph1.as_default():
                assign_weights = []
                for j in range(len(gen1_weights)):
                    assign_weights.append(tf.assign(gan1.gen_vars[j], gen1_weights[j]))
                for assignment in assign_weights:
                    sess1.run(assignment)
            print('Reset weights for generator 1')
    print('Scores computed')
    print(comparisons)
    for (model, scores) in comparisons.items():
        mu = np.mean(comparisons[model])
        sigma = np.std(comparisons[model])
        print('%s: mu = %f sigma = %f' % (model, mu, sigma))

    #
    #
    # images = data.next_batch(batch_size * num_disc)[0]
    # n_disc1 = 1
    # # f_name = 'mnist_%d_%s_%s' % (i, t, l)
    # f_name = 'mnist_1_256_drop_0_7'  # % i
    # print(f_name)
    # if not os.path.isfile('%s/%s/model.ckpt' % (FLAGS.dataset, f_name)):
    #     print('Model does not exist. Next')
    #     exit()
    #     # continue
    # # try:
    # graph1 = tf.Graph()
    # with graph1.as_default():
    #     gan1 = MultiDescGAN(num_latent, image_size, batch_size, n_disc1, num_channels=num_c,
    #                         num_hidden=128, mixing='arithmetic')
    #
    #     saver1 = tf.train.Saver()
    # # init1 = tf.initialize_variables(gan1.total_dis_vars + gan1.gen_vars)
    # sess1 = tf.Session(graph=graph1, config=config)
    # # sess1.run(init1)
    # saver1.restore(sess1, "%s/%s/model.ckpt" % (FLAGS.dataset, f_name))
    # # print('Managed to load first graph')
    #
    # # except Exception:
    # #     print('Could not load it. Continuing on')
    # #     exit()
    # #     # continue
    #
    # graph2 = tf.Graph()
    # # n_disc1 = 1
    # with graph2.as_default():
    #     gan2 = MultiDescGAN(num_latent, image_size, batch_size, 1, num_channels=num_c,
    #                         num_hidden=128, mixing='arithmetic')
    #     saver2 = tf.train.Saver()
    # # init2 = tf.initialize_variables(gan2.total_dis_vars + gan2.gen_vars)
    # sess2 = tf.Session(graph=graph2, config=config)
    # # sess2.run(init2)
    # saver2.restore(sess2, "mnist/mnist_1_256/model.ckpt")
    # # print('Managed to load the second one')
    #
    # # Compare performance on real images
    # # pred_1 = np.mean(sess1.run(gan1.losses, feed_dict={gan1.reals: images, gan1.training: True, gan1.l:0.}))
    # # gen_loss1 = sess1.run(gan1.gen_loss, feed_dict={gan1.training: True, gan1.l: 0.})
    # # # print('L(D1(G1)): %0.3f' % pred_1)
    # # pred_2 = np.mean(sess2.run(gan2.losses, feed_dict={gan2.reals: images, gan2.training: True, gan2.l: 0.}))
    # # gen_loss2 = sess2.run(gan2.gen_loss, feed_dict={gan2.training: True, gan2.l: 0.})
    # # # print('L(D2(G2)): %0.3f' % pred_2)
    # # ratio = pred_1 / pred_2
    # # print('Desc ratio: %0.4f' % ratio)
    #
    # # Pre generator swap
    # s11 = sess1.run(gan1.mixed_v, feed_dict={gan1.reals: images, gan1.training: True, gan1.l: 0.0})
    # s22 = sess2.run(gan2.mixed_v, feed_dict={gan2.reals: images[:100], gan2.training: True, gan2.l: 0.})
    # print('s11: %0.3f' % s11)
    # print('s22: %0.3f' % s22)
    #
    # # Swap generators:
    # gen1_weights = sess1.run(gan1.gen_vars)
    # gen2_weights = sess2.run(gan2.gen_vars)
    # with graph1.as_default():
    #     assign_weights = []
    #     for j in range(len(gen1_weights)):
    #         assign_weights.append(tf.assign(gan1.gen_vars[j], gen2_weights[j]))
    #     for assignment in assign_weights:
    #         sess1.run(assignment)
    # # print('Swapped out weights for generator 1')
    # with graph2.as_default():
    #     assign_weights = []
    #     for j in range(len(gen2_weights)):
    #         assign_weights.append(tf.assign(gan2.gen_vars[j], gen1_weights[j]))
    #     for assignment in assign_weights:
    #         sess2.run(assignment)
    # # print('Swapped out weights for generator 2')
    #
    # # gpred_1 = sess1.run(gan1.gen_loss, feed_dict={gan1.training: True, gan1.l: 0.})
    # # # np.mean(sess1.run(gan1.losses, feed_dict={gan1.reals: images, gan1.training: True}))
    # # # print('L(D1(G2)): %0.3f' % gpred_1)
    # # gpred_2 = sess2.run(gan2.gen_loss, feed_dict={gan2.training: True, gan2.l: 0.})
    # # # np.mean(sess2.run(gan2.losses, feed_dict={gan2.reals: images, gan2.training: True}))
    # # # print('L(D2(G1)): %0.3f' % gpred_2)
    #
    # # Post Generator Swap
    # s12 = sess1.run(gan1.mixed_v, feed_dict={gan1.reals: images, gan1.training: True, gan1.l: 0.0})
    # s21 = sess2.run(gan2.mixed_v, feed_dict={gan2.reals: images[:100], gan2.training: True, gan2.l: 0.})
    # print('s12: %0.3f' % s12)
    # print('s21: %0.3f' % s21)
    # score = (s12 / s11) / (s21 / s22)
    # # score = (gpred_1 / gen_loss1) / (gpred_2 / gen_loss2)
    # print('Score: %0.4f' % score)


if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_string("dataset", "cifar", "The name of dataset [celebA, mnist, cifar]")
    flags.DEFINE_integer("batch_size", 100, "The size of batch images [100]")
    flags.DEFINE_integer("latent", 100, "number of latent variables. [100]")
    flags.DEFINE_string("path", "cifar", "The name of directory where all the models are stored [celebA, mnist, cifar]")
    FLAGS = flags.FLAGS
    tf.app.run()
