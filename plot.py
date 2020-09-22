import json
import pandas as pd
import glob
import os.path
import logging
from plotnine import *

logging.basicConfig(level=logging.INFO)


def load_results(d, fname='result.json'):
    return [json.loads(line) for line in open("{}/{}".format(d, fname))]


def select(path, spec):
    data = json.load(open(path))
    return [x for x in data if all(x.get(key, False) == val for key, val in spec.items())][0]


def select_all(path, spec):
    data = json.load(open(path))
    return [x for x in data if all(x.get(key, False) == val for key, val in spec.items())]


def by_size():
    for d in glob.glob("experiments/vq-*"):
        logging.info("Loading results from {}".format(d))
        size = d.split('-')[1]
        level = d.split('-')[2][1]
        plus = '+' in d
        cors = select_all("{}/ed_rsa_fragments.json".format(d), dict(model='trained', reference='phoneme', by_size=True))
        for cor in cors:

            trained = dict(condition=os.path.basename(d),
                           plus=plus,
                           mode='trained',
                           size=size,
                           level=level,
                           cor=cor['cor'],
                           reference='phoneme',
                           quantile=cor['quantile'])
            yield trained


def scores():
    for d in glob.glob("experiments/vq-*"):
        try:
            logging.info("Loading results from {}".format(d))
            size = d.split('-')[1]
            level = d.split('-')[2][1]

            ret = sorted(load_results(d), key=lambda x: x['recall']['10'])[-1]
            ret_base = load_results(d)[0]
            zs = json.load(open("{}/vq_result.json".format(d)))
            zs_base = json.load(open("{}/vq_base_result.json".format(d)))

            cor      = select("{}/ed_rsa.json".format(d), dict(model='trained', reference='phoneme', by_size=False))['cor']
            cor3     = select("{}/ed_rsa_trigrams.json".format(d), dict(model='trained', reference='phoneme', by_size=False))['cor']
            corw1    = select("{}/ed_rsa_wordgrams1.json".format(d), dict(model='trained', reference='phoneme', by_size=False))['cor']
            corw5    = select("{}/ed_rsa_wordgrams5.json".format(d), dict(model='trained', reference='phoneme', by_size=False))['cor']
            corF     = select("{}/ed_rsa_fragments.json".format(d), dict(model='trained', reference='phoneme', by_size=False))['cor']
            cor_base = select("{}/ed_rsa.json".format(d), dict(model='random', reference='phoneme', by_size=False))['cor']
            cor_word = select("{}/ed_rsa.json".format(d), dict(model='trained', reference='word', by_size=False))['cor']
            cor_word_base = select("{}/ed_rsa.json".format(d), dict(model='random', reference='word', by_size=False))['cor']
            diag     = select("{}/local/local_diagnostic.json".format(d), dict(model='trained'))['acc']
            diag_base = select("{}/local/local_diagnostic.json".format(d), dict(model='random'))['acc']

            trained = dict(
                condition=os.path.basename(d),
                mode='trained',
                size=size,
                level=level,
                epoch=ret['epoch'],
                recall=ret['recall']['10'],
                abx=100-zs['2019']['english']['scores']['abx'],
                abx_lev=100-zs['2019']['english']['details_abx']['test']['levenshtein'],
                abx_f=100-json.load(open("{}/abx_flickr8k_result.json".format(d)))['avg_abx_error'],
                abx_fw=100-json.load(open("{}/abx_within_flickr8k_result.json".format(d)))['avg_abx_error'],
                abx_fr=100-json.load(open("{}/flickr8k_abx_rep_result.json".format(d)))['avg_abx_error'],
                abx_frw=100-json.load(open("{}/flickr8k_abx_rep_within_result.json".format(d)))['avg_abx_error'],
                bitrate=zs['2019']['english']['scores']['bitrate'],
                rle_ratio=json.load(open("{}/rle_compression.json".format(d)))['ratio'],
                ed_rsa=cor,
                ed_rsa3=cor3,
                ed_rsaw1=corw1,
                ed_rsaw5=corw5,
                ed_rsa_F=corF,
                ed_rsa_word=cor_word,
                recall_diff=ret['recall']['10'] - ret_base['recall']['10'],
                abx_diff=(100-zs['2019']['english']['scores']['abx']) - (100-zs_base['2019']['english']['scores']['abx']),
                ed_rsa_diff=cor - cor_base,
                ed_rsa_word_diff=cor_word - cor_word_base,
                diag=diag,
                diag_diff=diag-diag_base)
        except FileNotFoundError as e:
            logging.warning("MISSING DATA FOR {}\\{}".format(d, e))

        yield trained


def dump():
    #data = pd.read_json(json.dumps(list(scores())), orient='records')
    #data.to_csv("vq_experiment_stats.csv", header=True, index=False)

    data_by_size = pd.read_json(json.dumps(list(by_size())), orient='records')
    data_by_size.to_csv('by_size.csv', header=True, index=False)


def from_records(rows):
    return pd.read_json(json.dumps(list(rows)), orient='records')


def sizewise(rows):
    records = []
    for row in rows:
        common = dict(condition=row['condition'], codebook=row['size'], level=row['level'], epoch=row['epoch'],
                      recall=row['recall'], bitrate=row['bitrate'], rle_ratio=row['rle_ratio'], plus='+' in row['condition'])
        records.append({'size': '10', 'cor': row['ed_rsa'], **common})
        records.append({'size': '0', 'cor': row['ed_rsa3'], **common})
        records.append({'size': '1', 'cor': row['ed_rsaw1'], **common})
        records.append({'size': '5', 'cor': row['ed_rsaw5'], **common})
    data = from_records(records)

    g = ggplot(data.query('level==1 & plus == False & size in [1,5]'), aes(x='size', y='cor', color='factor(codebook)')) +\
                                                              geom_point() +\
                                                              geom_line() +\
                                                              xlab("Fragment size in words") +\
                                                              ylab("RSA correlation")

    ggsave(g, 'sizewise.pdf')
    g = ggplot(data.query('level==1 & plus == False & size in [0,10]'), aes(x='size', y='cor', color='factor(codebook)')) +\
                                                              geom_point() +\
                                                              geom_line() +\
                                                              xlab("Input size") +\
                                                              ylab("RSA correlation") +\
                                                              scale_x_continuous(breaks=[0, 10], labels=["3 phonemes","complete"])

    ggsave(g, 'sizewise2.pdf')


def r2_partial():
    data = pd.read_csv("partial_rsa_by_size-text.csv")
    g = ggplot(data, aes(x='size', y='r2', color='factor(codebook)')) + geom_point() + geom_line() + \
                            labs(color='codebook size', y='$R^2$')
    ggsave(g, "plot-r2.pdf")
    g = ggplot(data, aes(x='size', y='r_xonly', color='factor(codebook)')) + geom_point() + geom_line() + \
                            labs(color='codebook size', y="Pearson's r")
    ggsave(g, "plot-r.pdf")
    g = ggplot(data, aes(x='size', y='r2_part', color='factor(codebook)')) + geom_point() + geom_line() + \
                                                        labs(color='codebook', y='$R^2$ partial')
    ggsave(g, "plot-r2_partial-text.pdf")

    d1 = data[['size', 'codebook', 'r_xonly']].rename(columns={'r_xonly': 'r'})
    d2 = data[['size', 'codebook', 'r_base']].rename(columns={'r_base': 'r'})
    d1['reference'] = 'phonemes'
    d2['reference'] = 'vis-sem'
    data = d1.merge(d2, how='outer')

    g = ggplot(data, aes(x='size', y='r', color='factor(codebook)', linetype='reference')) + geom_point() + geom_line() + \
                                                        labs(color='codebook', y="Pearson's r")
    ggsave(g, "plot-r-control-text.pdf")

    data = pd.read_csv("partial_rsa_by_size-vis.csv")
    g = ggplot(data, aes(x='size', y='r2_part', color='factor(codebook)')) + geom_point() + geom_line() + \
                                                        labs(color='codebook', y='$R^2$ partial')
    ggsave(g, "plot-r2_partial-vis.pdf")


def compare_rsa_plot():
    rows = []
    base = "experiments/basic-stack/mean/global_rsa.json"
    row = select(base, dict(model='trained', layer='rnn1'))
    row['size'] = None
    rows.append(row)
    sizes = [2**n for n in range(5, 11) ]
    for size in sizes:
        try:
            row = select("experiments/vq-{}-q1/mean/global_rsa.json".format(size), dict(model='trained', layer='rnn_top0'))
            row['size'] =size
            rows.append(row)
        except:
            pass
    data = pd.DataFrame.from_records(rows)
    data.to_csv("compare_rsa.csv", header=True, index=False)
    baseline = data.query("size!=size")
    g = ggplot(data.query('size==size'), aes(x='size', y='cor'))  + geom_point() + geom_line() + scale_x_continuous(trans='log2') + \
                                                labs(x='codebook size', y="Pearson's r") + \
                                                geom_hline(data=baseline, mapping=aes(yintercept='cor'), linetype='dashed', show_legend=True)
    ggsave(g, "compare_rsa.pdf")


def main():
    rows = list(scores())
    sizewise(rows)
    data = from_records(rows)
    #vars = ['abx', 'abx_lev', 'abx_f', 'abx_fw', 'abx_fr', 'abx_frw', 'ed_rsa', 'ed_rsa_word', 'diag', 'ed_rsa3', 'ed_rsa_F']
    vars = ['abx_fw', 'ed_rsa', 'ed_rsa3',  'ed_rsa_word', 'diag']
    for var in vars:

        p = ggplot(data, aes(x='recall', y=var)) + \
                                geom_point(aes(size='bitrate', shape='factor(level)', color='factor(size)'))
        ggsave(p, 'plot-recall-{}.pdf'.format(var))

        for var2 in vars:
            if var < var2:
                p = ggplot(data, aes(x=var, y=var2)) + \
                                        geom_point(aes(size='bitrate', shape='factor(level)', color='factor(size)'))
                ggsave(p, 'plot-{}-{}.pdf'.format(var, var2))
