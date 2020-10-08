from collections import Counter
import glob
from itertools import groupby
import json
from math import log2
import numpy as np
import os.path
from pathlib import Path
import pickle
import plotnine as pn
from scipy.stats import entropy
import sklearn.metrics as M

import platalea.dataset as dataset
from plot import from_records, select
import prepare_flickr8k as pf


def count_repetitions(array):
    return [len(list(v)) for _, v in groupby(array)]


def proportion_top1(labels, codes):
    start = segments_start(labels)
    segments = split_sequences(codes, start)
    return [sorted(Counter(s).values())[-1] / len(s) for s in segments]


def flatten(xss):
    return [x for xs in xss for x in xs]


def split_sequences(array, start):
    end = start[1:] + [len(array)]
    return [array[s:e] for s, e in zip(start, end)]


def segments_start(array):
    return [i for i in range(len(array)) if i == 0 or array[i] != array[i-1]]


def compute_general_statistics(directory='.'):
    stats = []
    with open('{}/local_trained_codebook.pkl'.format(directory), 'rb') as f:
        local = pickle.load(f)
    num_frames = len(local['codebook']['features'])
    with open('{}/global_input.pkl'.format(directory), 'rb') as f:
        global_input = pickle.load(f)
    words = [x.lower().split() for x in global_input['text']]
    phones = global_input['ipa']
    for level, data in [('words', words), ('phones', phones)]:
        num_tokens = np.sum([len(d) for d in data]).item()
        num_uniq_values = len(set(flatten(data)))
        num_mean_rep = num_frames / num_tokens
        stats.append(dict(
            level=level,
            num_tokens=num_tokens,
            num_uniq_values=num_uniq_values,
            num_mean_rep=num_mean_rep))
    return stats


def compute_joint_probability(x, y):
    labels_x = np.unique(x)
    idx_x = {v: i for i, v in enumerate(labels_x)}
    labels_y = np.unique(y)
    idx_y = {v: i for i, v in enumerate(labels_y)}
    counts_xy = np.zeros([len(labels_x), len(labels_y)])
    for xi, yi in zip(x, y):
        counts_xy[idx_x[xi], idx_y[yi]] += 1
    return labels_x, labels_y, counts_xy / len(x)


def conditional_entropy(x, y):
    labels_x, labels_y, p_xy = compute_joint_probability(x, y)
    p_y = np.sum(p_xy, axis=0)
    h_x_y = 0
    for i_x in range(len(labels_x)):
        for i_y in range(len(labels_y)):
            if p_xy[i_x, i_y] > 0:
                h_x_y -= p_xy[i_x, i_y] * log2(p_xy[i_x, i_y] / p_y[i_y])
    return h_x_y


def compute_metrics(path):
    split = path.split('-')
    size = int(split[1])
    level = int(split[2][1])
    d = Path(path)
    D = pickle.load(open(d / 'local_trained_codebook.pkl', 'rb'))
    labels = D['codebook']['labels']
    indices = [d.nonzero()[0][0] for d in D['codebook']['features']]
    D_w = pickle.load(open(d / 'global_trained_codebook.pkl', 'rb'))
    labels_w, activations_w = load_word_alignments(D_w['codebook'])
    indices_w = [a.nonzero()[0][0] for a in activations_w]
    cond_word = lambda w: len(w['phones']) > 2
    #labels_wl, activations_wl = load_word_alignments(D_w['codebook'], cond_word)
    #indices_wl = [a.nonzero()[0][0] for a in activations_wl]
    num_mean_rep = np.mean(count_repetitions(indices))
    prop_top_code = np.mean(proportion_top1(labels, indices))
    prop_top_label = np.mean(proportion_top1(indices, labels))
    prop_top_code_w = np.mean(proportion_top1(labels_w, indices_w))
    #prop_top_code_wl = np.mean(proportion_top1(labels_wl, indices_wl))
    prop_top_label_w = np.mean(proportion_top1(indices_w, labels_w))
    values, counts = np.unique(indices, return_counts=True)
    num_uniq_ind = len(values)
    entropy_ind = entropy(counts, base=2)
    cond_entropy_ind = conditional_entropy(indices, labels)
    cond_entropy_lab = conditional_entropy(labels, indices)
    mutual_info = entropy_ind - cond_entropy_ind
    ami = M.adjusted_mutual_info_score(indices, labels)
    ari = M.adjusted_rand_score(indices, labels)
    vmeasure = M.v_measure_score(indices, labels)
    rsa = select(
        "{}/ed_rsa.json".format(d),
        dict(model='trained', reference='phoneme', by_size=False))['cor']
    rsaF = select(
        "{}/ed_rsa_fragments.json".format(d),
        dict(model='trained', reference='phoneme', by_size=False))['cor']
    rsa3 = select(
        "{}/ed_rsa_trigrams.json".format(d),
        dict(model='trained', reference='phoneme', by_size=False))['cor']
    diag = select(
        "{}/local/local_diagnostic.json".format(d),
        dict(model='trained'))['acc']
    abx_fw = 100 - json.load(
        open("{}/abx_within_flickr8k_result.json".format(d)))['avg_abx_error']
    abx_frw = 100 - json.load(
        open("{}/flickr8k_abx_rep_within_result.json".format(d)))['avg_abx_error']
    return dict(
        size=size,
        level=level,
        num_mean_rep=num_mean_rep,
        prop_top_code=prop_top_code,
        prop_top_label=prop_top_label,
        prop_top_code_w=prop_top_code_w,
        #prop_top_code_wl=prop_top_code_wl,
        prop_top_label_w=prop_top_label_w,
        num_uniq_ind=num_uniq_ind,
        entropy_ind=entropy_ind,
        cond_entropy_ind=cond_entropy_ind,
        cond_entropy_lab=cond_entropy_lab,
        mutual_info=mutual_info,
        ami=ami,
        ari=ari,
        vmeasure=vmeasure,
        norm_entropy_ind=entropy_ind / log2(size),
        norm_cond_entropy_ind=cond_entropy_ind / log2(size),
        norm_cond_entropy_lab=cond_entropy_lab / log2(40),
        norm_entropy_ind2=entropy_ind / log2(num_uniq_ind),
        norm_cond_entropy_ind2=cond_entropy_ind / log2(num_uniq_ind),
        rsa=rsa,
        rsaF=rsaF,
        rsa3=rsa3,
        diag=diag,
        abx_fw=abx_fw,
        abx_frw=abx_frw)


def load_word_alignments(activations, cond_word=None):
    data = pf.load_alignment('data/datasets/flickr8k/fa.json')
    val = dataset.Flickr8KData(root='data/datasets/flickr8k/', split='val',
                               feature_fname='mfcc_features.pt')
    # Vocabulary should be initialized even if we are not going to use text
    # data
    if dataset.Flickr8KData.le is None:
        dataset.Flickr8KData.init_vocabulary(val)
    alignments = [data[sent['audio_id']] for sent in val]
    # Only consider cases where alignement does not fail
    alignments = [item for item in alignments if pf.good_alignment(item)]
    factors = pf.default_factors()
    index = pf.make_indexer(factors, 'codebook')
    labels = []
    states = []
    for act, ali in zip(activations, alignments):
        # extract word labels for current utterance
        fr = list(word_frames(ali, act, index, cond_word))
        if len(fr) > 0:
            y, X = zip(*fr)
            y = np.array(y)
            X = np.stack(X)
            labels.append(y)
            states.append(X)
    return np.concatenate(labels), np.concatenate(states)


def word_frames(utt, rep, index, cond_word=None):
    """
    Return a sequence of pairs (word label, frame), given an alignment object
    `utt`, a representation array `rep`, and indexing function `index`.
    """
    for w, start, end in words(utt, cond_word):
        assert index(start) < index(end)+1, "Something funny: {} {} {} {}".format(start, end, index(start), index(end))
        for j in range(index(start), index(end)+1):
            if j < rep.shape[0]:
                yield (w, rep[j])
            else:
                logging.warning("Index out of bounds: {} {}".format(j, rep.shape))


def words(utt, cond=None):
    """
    Return sequence of words labels associated with start and end time
    corresponding to the alignment JSON object `utt`.
    """
    for word in utt['words']:
        label = word['word']
        if label != 'oov' and (cond is None or cond(word)):
            yield (label, int(word['start']*1000), int(word['end']*1000))


def compute_and_save_metrics():
    results = []
    path_tplt = '/roaming/gchrupal/verdigris/platalea.vq/experiments/vq-*'
    for i, path in enumerate(glob.glob(path_tplt)):
        print('Working on experiment {}'.format(path))
        d = Path(path)
        if i == 0:
            stats = compute_general_statistics(path)
            stats = [list(s.values())[1:] for s in stats]
            np.savetxt('exploration.csv', stats, delimiter=',')
        try:
            metrics = compute_metrics(path)
        except FileNotFoundError:
            print('!!!Skipping experiment {}'.format(path))
            continue
        results.append(list(metrics.values()))
        expname = d.parts[-1]
        path_out = Path('experiments/{}'.format(expname))
        path_out.mkdir(parents=True, exist_ok=True)
        json.dump(metrics, open(path_out / 'entropy.json', 'w'))
    with open('exploration.csv', 'ab') as f:
        np.savetxt(f, results, delimiter=',')


def read_metrics():
    metrics = []
    path_tplt = '/roaming/gchrupal/verdigris/platalea.vq/experiments/vq-*'
    for i, path in enumerate(glob.glob(path_tplt)):
        d = Path(path)
        expname = d.parts[-1]
        path_out = Path('experiments/{}'.format(expname))
        path_out.mkdir(parents=True, exist_ok=True)
        metrics.append(json.load(open(path_out / 'entropy.json')))
    return metrics


def plot_distributions():
    distr_ind = []
    distr_lab = []
    for p in ['/roaming/gchrupal/verdigris/platalea.vq/experiments/vq-32-q1',
              '/roaming/gchrupal/verdigris/platalea.vq/experiments/vq-1024-q1',
              '/roaming/gchrupal/verdigris/platalea.vq/experiments/vq-32-q2',
              '/roaming/gchrupal/verdigris/platalea.vq/experiments/vq-1024-q2',
              '/roaming/gchrupal/verdigris/platalea.vq/experiments/vq-32-q3',
              '/roaming/gchrupal/verdigris/platalea.vq/experiments/vq-1024-q3']:
        d_ind, d_lab = sample_distributions(p)
        distr_ind.extend(d_ind)
        distr_lab.extend(d_lab)
    distr_ind = from_records(distr_ind)
    distr_lab = from_records(distr_lab)
    p = pn.ggplot(data=distr_ind) +\
        pn.geom_bar(mapping=pn.aes(x="index", y="probability"),
                    stat="identity") +\
        pn.facet_grid("size ~ level")
    pn.ggsave(p, 'plot-distr-ind.pdf')
    p = pn.ggplot(data=distr_lab) +\
        pn.geom_bar(mapping=pn.aes(x="index", y="probability"),
                    stat="identity") +\
        pn.facet_grid("size ~ level")
    pn.ggsave(p, 'plot-distr-lab.pdf')


def sample_distributions(path):
    d = Path(path)
    condition = os.path.basename(d)
    split = condition.split('-')
    size = int(split[1])
    level = int(split[2][1])
    D = pickle.load(open(d / 'local_trained_codebook.pkl', 'rb'))
    indices = [d.nonzero()[0][0] for d in D['codebook']['features']]
    labels = D['codebook']['labels']
    _, _, p_xy = compute_joint_probability(indices, labels)
    distr_ind = sorted(p_xy[0, :] / np.sum(p_xy[0, :]), reverse=True)
    distr_ind = [{'index': i,
                  'probability': p,
                  'size': size,
                  'level': level} for i, p in enumerate(distr_ind)]
    distr_lab = sorted(p_xy[:, 0] / np.sum(p_xy[:, 0]), reverse=True)
    distr_lab = [{'index': i,
                  'probability': p,
                  'size': size,
                  'level': level} for i, p in enumerate(distr_lab)]
    return distr_ind, distr_lab


def plot_distributions_3d():
    for p in ['/roaming/gchrupal/verdigris/platalea.vq/experiments/vq-32-q1',
              '/roaming/gchrupal/verdigris/platalea.vq/experiments/vq-1024-q1',
              '/roaming/gchrupal/verdigris/platalea.vq/experiments/vq-32-q2',
              '/roaming/gchrupal/verdigris/platalea.vq/experiments/vq-1024-q2',
              '/roaming/gchrupal/verdigris/platalea.vq/experiments/vq-32-q3',
              '/roaming/gchrupal/verdigris/platalea.vq/experiments/vq-1024-q3']:
        condition = os.path.basename(p)
        split = condition.split('-')
        size = int(split[1])
        level = int(split[2][1])
        d_ind, d_lab = compute_distributions(p)
        plot_3d(d_ind, 'plot-distr-ind-{}-{}.pdf'.format(level, size))
        plot_3d(d_lab.T, 'plot-distr-lab-{}-{}.pdf'.format(level, size))


def compute_distributions(path):
    d = Path(path)
    D = pickle.load(open(d / 'local_trained_codebook.pkl', 'rb'))
    indices = [d.nonzero()[0][0] for d in D['codebook']['features']]
    labels = D['codebook']['labels']
    _, _, p_xy = compute_joint_probability(indices, labels)
    p_xy_sy = np.array(p_xy)
    for i_x in range(p_xy.shape[0]):
        p_xy_sy[i_x, :] = sorted(p_xy[i_x, :], reverse=True)
        p_xy_sy[i_x, :] = p_xy_sy[i_x, :] / np.sum(p_xy[i_x, :])
    p_xy_sx = np.array(p_xy)
    for i_y in range(p_xy.shape[1]):
        p_xy_sx[:, i_y] = sorted(p_xy[:, i_y], reverse=True)
        p_xy_sx[:, i_y] = p_xy_sx[:, i_y] / np.sum(p_xy[:, i_y])
    return p_xy_sy, p_xy_sx


def plot_3d(data, fname):
    import numpy as np
    import matplotlib.pyplot as plt
    # This import registers the 3D projection, but is otherwise unused.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

    # setup the figure and axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    _x = np.arange(data.shape[1])
    _y = np.arange(data.shape[0])
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    top = data.ravel()
    bottom = np.zeros_like(top)
    width = depth = 1

    ax.bar3d(x, y, bottom, width, depth, top, shade=True)
    plt.tight_layout()
    plt.savefig(fname)


def plot():
    data = read_metrics()
    data = from_records(data)
    for v in ['num_uniq_ind', 'norm_cond_entropy_ind2']:
        p = pn.ggplot(data, pn.aes(x='factor(size)', y=v,
                                   shape='factor(level)')) +\
            pn.geom_point()
        pn.ggsave(p, 'plot-{}.pdf'.format(v))
    for v, var2 in [('vmeasure', ['ami', 'rsa', 'diag']),
                    ('abx_fw', ['rsa', 'rsaF', 'rsa3']),
                    ('abx_frw', ['rsa', 'abx_fw', 'diag']),
                    ('rsaF', ['rsa', 'diag'])]:
        for v2 in var2:
            p = pn.ggplot(data, pn.aes(x=v, y=v2)) + \
                pn.geom_point(pn.aes(shape='factor(level)',
                                     color='factor(size)'))
            pn.ggsave(p, 'plot-{}-{}.pdf'.format(v, v2))
    path = '/roaming/gchrupal/verdigris/platalea.vq/experiments/vq-32-q1'
    stats = compute_general_statistics(path)
    stats = from_records(stats)
    p = pn.ggplot(data, pn.aes(x='factor(level)', y='num_mean_rep')) +\
        pn.geom_point(pn.aes(color='factor(size)')) +\
        pn.geom_hline(stats, pn.aes(yintercept='num_mean_rep',
                                    linetype='level'))
    pn.ggsave(p, 'plot-num_mean_rep.pdf')
    for v in ['prop_top_code', 'prop_top_code_w', 'prop_top_label',
              'prop_top_label_w']:
        p = pn.ggplot(data, pn.aes(x='factor(level)', y=v)) +\
            pn.geom_point(pn.aes(color='factor(size)'))
        pn.ggsave(p, 'plot-{}.pdf'.format(v))
