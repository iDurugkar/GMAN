import matplotlib
import tensorflow as tf
import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(font_scale=2.2)
# sns.set(sns.axes_style('white'), rc={'axes.facecolor': 'white', 'figure.facecolor':'grey',
#                                      'axes.linecolor': 'grey'})
import numpy as np
from os import listdir
from os.path import isfile, join
matplotlib.rcParams.update({'font.size': 22})

i = 0
x2 = []
temp = []
x1 = []
x3 = []
temp2 = []

step = 20

for ind in range(1, 5):
    files = [f for f in listdir('mnist_before_2nd/1_original_%d' % ind) if f.startswith('events')]
    fname = files[-1]
    for summary in\
            tf.train.summary_iterator(join('mnist_before_2nd/1_original_%d' % ind, fname)):
        # ("mnist/mnist_2_geometric_0.0/events.out.tfevents.1477169818.manifold"):
        # print(summary)
        try:
            for s in summary.summary.value:
                if s.tag.startswith('gen_loss'):
                    temp.append(s.simple_value)
                    i += 1
                    if i == step:
                        temp2.append(np.mean(temp))
                        temp = []
                        i = 0
        except Exception:
            print('not iterable')
    x1.append(temp2)
    temp2 = []
    # i += 1
    # if i > step:
    #     exit()
i = 0
temp = []

for ind in range(1, 5):
    files = [f for f in listdir('mnist/2_self_%d' % ind) if f.startswith('events')]  # harmonic_0.
    fname = files[-1]
    for summary in\
            tf.train.summary_iterator(join('mnist/2_self_%d' % ind, fname)):
        # ("mnist/mnist_1/events.out.tfevents.1477248849.manifold"):
        # print(summary)
        try:
            for s in summary.summary.value:
                if s.tag.startswith('gen_loss'):
                    temp.append(s.simple_value)
                    i += 1
                    if i == step:
                        temp2.append(np.mean(temp))
                        temp = []
                        i = 0
        except Exception:
            print('not iterable')
    x2.append(temp2)
    # print(temp2)
    temp2 = []
i = 0
temp = []

for ind in range(1, 3):
    files = [f for f in listdir('mnist/5_self_%d' % ind) if f.startswith('events')]  # 5_harmonic_0._
    fname = files[-1]
    print(fname)
    for summary in\
        tf.train.summary_iterator(join('mnist/5_self_%d' % ind, fname)):
        # ("mnist/mnist_5_geometric_0.0/events.out.tfevents.1477175657.manifold"):
        # print(summary)
        try:
            for s in summary.summary.value:
                if s.tag.startswith('gen_loss'):
                    temp.append(s.simple_value)
                    i += 1
                    if i == step:
                        temp2.append(np.mean(temp))
                        temp = []
                        i = 0
        except Exception:
            print('not iterable')
    x3.append(temp2)
    temp2 = []
x1 = np.asarray(x1)
x2 = np.asarray(x2)
x3 = np.asarray(x3)
# from IPython import embed
# embed()
print(x3.shape)
s = np.min([x1.shape[1], x2.shape[1], x3.shape[1]])
x1 = x1[:, :s]
x2 = x2[:, :s]
x3 = x3[:, :s]

x1_mean = np.mean(x1[1:], axis=0)
x1_std = np.std(x1[1:], axis=0)
# t = np.arange(x1_mean.shape[0])
t = np.arange(step, (x1_mean.shape[0] + 1)*step, step)
# print(x1_mean.shape)
# print(x1_std)
# print(x1.shape)
# print(x2.shape)
# print(x3.shape)

x2_mean = np.mean(x2, axis=0)
x2_std = np.std(x2, axis=0)

x3_mean = np.mean(x3, axis=0)
x3_std = np.std(x3, axis=0)

plt.plot(t,x1_mean, 'b-')
plt.plot(t,x2_mean, 'g-')
plt.plot(t,x3_mean, 'm-')
plt.fill_between(t, x1_mean + x1_std, x1_mean - x1_std, facecolor='blue', alpha=0.2)
plt.fill_between(t, x2_mean + x2_std, x2_mean - x2_std, facecolor='green', alpha=0.5)
plt.fill_between(t, x3_mean + x3_std, x3_mean - x3_std, facecolor='magenta', alpha=0.5)
# plt.title('Effect of multiple Discriminators on ')
plt.legend([r'$N=1$', r'$N=2$', r'$N=5$'])
plt.xlabel('Iteration #')
plt.ylabel(r'$log(1 - D(G(z)))$')
plt.ylim([0.0, -2.])
plt.tight_layout()
plt.savefig('mnist/1_original_1/mnist_gen_loss')

# plot cumulative standard deviation
x1_cumstd = np.asarray([np.std(x1[:,tt-500/step:tt],axis=1) for tt in range(int(500/step),x1.shape[1])]).T
x2_cumstd = np.asarray([np.std(x2[:,tt-500/step:tt],axis=1) for tt in range(int(500/step),x2.shape[1])]).T
x3_cumstd = np.asarray([np.std(x3[:,tt-500/step:tt],axis=1) for tt in range(int(500/step),x3.shape[1])]).T
# import IPython as ipy
# ipy.embed()
x1_cumstd_mean = x1_cumstd.mean(axis=0)
x2_cumstd_mean = x2_cumstd.mean(axis=0)
x3_cumstd_mean = x3_cumstd.mean(axis=0)
x1_cumstd_std = x1_cumstd.std(axis=0)
x2_cumstd_std = x2_cumstd.std(axis=0)
x3_cumstd_std = x3_cumstd.std(axis=0)
t = t[int(500/step):]
print(t.shape)
print(x1_cumstd_mean.shape)
plt.cla()
plt.clf()
plt.semilogy(t,x1_cumstd_mean, 'b-')
plt.semilogy(t,x2_cumstd_mean, 'g-')
plt.semilogy(t,x3_cumstd_mean, 'm-')
plt.semilogy(t, np.ones_like(t) * 1e-2, 'k--')
# plt.fill_between(t, x1_cumstd_mean + x1_cumstd_std, x1_cumstd_mean - x1_cumstd_std, facecolor='blue', alpha=0.2)
# plt.fill_between(t, x2_cumstd_mean + x2_cumstd_std, x2_cumstd_mean - x2_cumstd_std, facecolor='green', alpha=0.5)
# plt.fill_between(t, x3_cumstd_mean + x3_cumstd_std, x3_cumstd_mean - x3_cumstd_std, facecolor='magenta', alpha=0.5)
# plt.title('Effect of multiple Discriminators on ')
plt.legend([r'$N=1$', r'$N=2$', r'$N=5$'])
plt.xlabel('Iteration #')
plt.ylabel(r'Cumulative STD of $log(1 - D(G(z)))$')
# plt.ylim([0.0, -4.])
plt.tight_layout()
plt.savefig('mnist/1_original_1/mnist_gen_loss_std')


# x2_mean = np.mean(x2, axis=0)
# x2_std = np.std(x2, axis=0)
# x3_mean = np.mean(x3, axis=0)
# x3_std = np.std(x3, axis=0)
# t = np.arange(step, (x3_mean.shape[0] + 1)*step, step)
# plt.plot(t, x2_mean, 'g-')
# plt.fill_between(t, x2_mean + x2_std, x2_mean - x2_std, facecolor='green', alpha=0.5)
# plt.plot(t, x3_mean, 'm-')
# plt.fill_between(t, x3_mean + x3_std, x3_mean - x3_std, facecolor='magenta', alpha=0.5)
# plt.xlabel('Iteration #')
# plt.ylabel(r'$\lambda$')
# plt.legend([r'$N=2$', r'$N=5$'])
# plt.tight_layout()
# plt.savefig('mnist/1_original_1/learnt_lambda')
