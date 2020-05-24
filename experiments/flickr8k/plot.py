import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

plt.style.use('seaborn-darkgrid')

from utils.get_best_score import get_best_score


def extract_matched_results(root, pattern, path2type_fn):
    comp_ptrn = re.compile(pattern)
    res = {}
    for d in Path(root).iterdir():
        matched = comp_ptrn.match(str(d.name))
        if matched is not None:
            tmp_res = res
            for m in matched.groups()[:-1]:
                tmp_res = tmp_res.setdefault(m, {})
            score = get_best_score(d / 'result.json', path2type_fn(d))
            tmp_res[matched.groups()[-1]] = score
    return res


def extract_results(root, exp_types, ds_factors, replids, path2type_fn,
                    tag=""):
    res = np.zeros([len(exp_types), len(ds_factors), len(replids)])
    for i, etype in enumerate(exp_types):
        for j, ds in enumerate(ds_factors):
            for k, rid in enumerate(replids):
                pattern = '{}{}-ds{}-{}-*'.format(etype, tag, ds, rid)
                paths = sorted(Path(root).glob(pattern))
                if len(paths) != 1:
                    msg = 'Pattern {} matches {} folders'
                    raise ValueError(msg.format(pattern, len(paths)))
                score = get_best_score(paths[0] / 'result.json',
                                       path2type_fn(paths[0]))
                res[i, j, k] = score
    return res


def path2type(path):
    exptype = {'asr': 'asr', 'basic': 'retrieval', 'mtl': 'mtl',
               'pip': 'retrieval', 'text': 'retrieval'}
    expname = Path(path).name.split('-')[0]
    return exptype[expname]


def path2type_jp(path):
    exptype = {'asr': 'slt', 'basic': 'retrieval', 'mtl': 'mtl',
               'pip': 'retrieval', 'text': 'retrieval'}
    expname = Path(path).name.split('-')[0]
    return exptype[expname]


def dict2np(res):
    if type(res) == dict:
        return [dict2np(v) for v in res.values()]
    else:
        return res


def plot_downsampling():
    # Extracting results
    basic_default_results = extract_matched_results(
        'runs', '(basic-default)-([abc])-.*', path2type)
    basic_default_score = np.mean(dict2np(basic_default_results))
    print("basic-default: {}".format(basic_default_score))
    exp_types = ['asr', 'text-image', 'pip-ind', 'pip-seq', 'mtl-asr', 'mtl-st']
    ds_factors = [1, 3, 9, 27, 81, 243]
    ds_factors_text = [str(i).zfill(3) for i in ds_factors]
    replids = ['a', 'b', 'c']
    res = extract_results('runs', exp_types, ds_factors_text, replids,
                          path2type)
    res = np.mean(res, axis=2)
    print(exp_types)
    print(res)

    # Plotting
    xticklabels = ['34 h', '11.3 h', '3.8 h', '1.3 h', '25 mins', '8 mins']
    fig, ax = plt.subplots()
    ax.set_xlabel('Amount of transcribed data available for training (total speech duration)')
    plt.xticks(range(len(xticklabels)), xticklabels, size='small')
    ax.set_ylabel('R@10')
    ax.plot([basic_default_score] * len(ds_factors), 'r--',
            label='speech-image')
    #ax.plot(res[0, :], '.--', label=exp_types[0])
    for i in range (1, len(exp_types)):
        ax.plot(res[i, :], '.-', label=exp_types[i])
    #ax2 = ax.twinx()
    #ax2.set_ylabel('WER (ASR experiments)')
    ax.legend()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    #plt.show()
    plt.savefig('downsampling_text.pdf')


def plot_downsampling_jp():
    # Extracting results
    basic_default_results = extract_matched_results(
        'runs', '(basic-default)-jp-([abc])-.*', path2type)
    basic_default_score = np.mean(dict2np(basic_default_results))
    print("basic-default: {}".format(basic_default_score))
    exp_types = ['asr', 'text-image', 'pip-ind', 'pip-seq', 'mtl-asr', 'mtl-st']
    exp_legend = ['slt', 'text-image', 'pip-ind', 'pip-seq', 'mtl-slt', 'mtl-st']
    ds_factors = [1, 3, 9, 27, 81, 243]
    ds_factors_text = [str(i).zfill(3) for i in ds_factors]
    replids = ['a', 'b', 'c']
    res = extract_results('runs', exp_types, ds_factors_text, replids,
                          path2type_jp, '-jp')
    res = np.mean(res, axis=2)
    print(exp_types)
    print(res)

    # Plotting
    fig, ax = plt.subplots()
    xticklabels = ['13.6 h', '4.5 h', '1.5 h', '30 mins', '10 mins', '3 min']
    ax.set_xlabel('Amount of translated data available for training (total speech duration)')
    plt.xticks(range(len(xticklabels)), xticklabels, size='small')
    ax.set_ylabel('R@10')
    ax.plot([basic_default_score] * len(ds_factors), 'r--',
            label='speech-image')
    for i in range (1, len(exp_types)):
        ax.plot(res[i, :], '.-', label=exp_legend[i])
    #ax2 = ax.twinx()
    #ax2.set_ylabel('BLEU score (SLT experiments)')
    ax.legend()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    #plt.show()
    plt.savefig('downsampling_text_jp.pdf')
