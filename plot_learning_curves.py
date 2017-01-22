import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 12})

import numpy as np

from os import listdir
from os.path import isfile, join

import tensorflow as tf


def get_summary_values(root, ind_range=range(1, 5), tag='G_loss', step=20):
    x = []
    min_len = np.inf

    for ind in ind_range:
        xind = []
        path = root + '%d' % ind
        files = [f for f in listdir(path) if f.startswith('events')]
        fname = files[-1]

        for summary in tf.train.summary_iterator(join(path, fname)):
            try:
                for s in summary.summary.value:
                    if s.tag.startswith(tag):
                        xind.append(s.simple_value)
            except Exception:
                print('Summary value not iterable')

        x.append(xind)
        min_len = min(min_len, len(xind))

    # set sequence length to shortest sequence
    for ind in range(len(x)):
        x[ind] = x[ind][:min_len]

    # smooth sequences with mean
    if step > 1:
        for ind in range(len(x)):
            xind_split = np.array_split(x[ind], min_len // step, axis=0)
            x[ind] = [np.mean(xi) for xi in xind_split]

    return np.asarray(x)


def get_means_stdevs(seqs, window_size=500, step=20):
    seq_tups = []
    # seq_tups_cumstd = []
    seq_tups_cumlogstd = []

    # set sequence length to shortest sequence and compute means/stdevs
    s = np.min([seq.shape[1] for seq in seqs])
    for seq_ind in range(len(seqs)):
        seq = seqs[seq_ind][:, :s]
        seq_mean = np.mean(seq, axis=0)
        seq_std = np.std(seq, axis=0)

        seq_tups += [(seq_mean, seq_std)]

        adj_win = int(window_size/step)
        seq_cumlogstd = np.log(np.asarray([np.std(seq[:,tt-adj_win:tt],axis=1) for tt in range(adj_win,s)]).T)
        seq_cumlogstd_mean = np.mean(seq_cumlogstd,axis=0)
        seq_cumlogstd_std = np.std(seq_cumlogstd,axis=0)

        seq_tups_cumlogstd += [(seq_cumlogstd_mean,seq_cumlogstd_std)]

    t = np.arange(step, (s + 1)*step, step)
    t_cumlogstd = t[adj_win:]

    return t, seq_tups, t_cumlogstd, seq_tups_cumlogstd


def make_plots(saveto_1, saveto_2, sum_configs, plt_configs):
    seqs = [get_summary_values(*sconf) for sconf in sum_configs]
    t, seq_tups, t_cumstd, seq_tups_cumstd = get_means_stdevs(seqs, step=sum_configs[0][-1])

    for sum_id, pconf in enumerate(plt_configs):
        color, linetyp, alpha, label = pconf
        seq_mean, seq_std = seq_tups[sum_id]
        plt.plot(t, seq_mean, color + linetyp, label=label)
        plt.fill_between(t, seq_mean - seq_std, seq_mean + seq_std, facecolor=color, alpha=alpha)

    plt.legend()
    plt.xlabel('Iteration #')
    plt.ylabel(r'$F(V(D,G))$')
    plt.ylim([-0.8, -0.2])
    plt.tight_layout()
    plt.savefig(saveto_1)

    plt.cla()
    plt.clf()

    for sum_id, pconf in enumerate(plt_configs):
        color, linetyp, alpha, label = pconf
        seq_mean, seq_std = seq_tups_cumstd[sum_id]
        plt.semilogy(t_cumstd,np.exp(seq_mean),color+linetyp,label=label)
        # plt.fill_between(t_cumstd, np.exp(seq_mean - seq_std), np.exp(seq_mean + seq_std), facecolor=color, alpha=alpha)

    plt.semilogy(t_cumstd, np.ones_like(t_cumstd) * 1e-2, 'k--')

    plt.legend()
    plt.xlabel('Iteration #')
    plt.ylabel(r'Cumulative STD of $F(V(D,G))$')
    plt.tight_layout()
    plt.savefig(saveto_2)


if __name__ == '__main__':
    sum_configs = []  # root, ind_range, tag, step
    plt_configs = []  # line/fill color, line type, alpha, legend label

    # Summary 1
    sum_configs += [('cifar/1_original_256_', range(1, 6), 'G_loss', 50)]
    plt_configs += [('b', '-', 0.2, r'$N=1$ Original')]

    # Summary 2
    sum_configs += [('cifar/1_modified_256_', range(1, 6), 'G_loss', 50)]
    plt_configs += [('g', '-', 0.5, r'$N=1$ Modified')]

    # Summary 3
    sum_configs += [('cifar/2_0_256_', range(2, 6), 'G_loss', 50)]
    plt_configs += [('m', '-', 0.5, r'$N=2$, $\lambda=0$')]

    # Summary 4
    sum_configs += [('cifar/2_1_256_', range(2, 6), 'G_loss', 50)]
    plt_configs += [('c', '-', 0.5, r'$N=2$, $\lambda=1$')]

    saveto_1 = 'cifar/cifar_gen_loss'
    saveto_2 = 'cifar/cifar_gen_loss_std'

    make_plots(saveto_1, saveto_2, sum_configs, plt_configs)
