import glob
import json
from pathlib import Path
from plot import from_records, select
import plotnine as pn


def extract_abx(fpath):
    if fpath.exists():
        return 100 - json.load(open(fpath))['avg_abx_error']
    else:
        return None


def extract_dc(fpath, mode):
    if fpath.exists():
        return select(fpath, dict(model=mode))['acc']
    else:
        return None


def extract_rsa(fpath, mode, ref):
    if fpath.exists():
        return select(fpath, dict(model=mode, reference=ref))['cor']
    else:
        return None


def extract_results_niekerk():
    results = []
    for path in glob.glob('experiments/niekerk/english/val/*/*'):
        d = Path(path)
        s = path.split('/')
        # Computing path for results with triplets
        s3 = s.copy()
        s3[2] = 'english_triplets'
        d3 = Path('/'.join(s3))
        results.append(dict(
            size=s[4],
            output_type=('embeddings' if s[5] == 'z' else 'indices'),
            dc_base=extract_dc(d / 'local/local_diagnostic.json', 'random'),
            dc=extract_dc(d / 'local/local_diagnostic.json', 'trained'),
            rsa_base=extract_rsa(d / 'ed_rsa.json', 'random', 'phoneme'),
            rsa_base_w=extract_rsa(d / 'ed_rsa.json', 'random', 'word'),
            rsa=extract_rsa(d / 'ed_rsa.json', 'trained', 'phoneme'),
            rsa_w=extract_rsa(d / 'ed_rsa.json', 'trained', 'word'),
            abx3_base=extract_abx(d3/ 'random/flickr8k_abx_within_result.json'),
            abx3=extract_abx(d3 / 'trained/flickr8k_abx_within_result.json'),
            rsa3_base=extract_rsa(d3 / 'ed_rsa_trigrams.json', 'random', 'phoneme'),
            rsa3_base_w=extract_rsa(d3 / 'ed_rsa_trigrams.json', 'random', 'word'),
            rsa3=extract_rsa(d3 / 'ed_rsa_trigrams.json', 'trained', 'phoneme'),
            rsa3_w=extract_rsa(d3 / 'ed_rsa_trigrams.json', 'trained', 'word')))
    return results


def plot_all():
    results = extract_results_niekerk()
    res = [r for r in results if r['abx3'] is not None and r['rsa3'] is not None]
    res = from_records(res)
    p = pn.ggplot(res, pn.aes(x='abx3', y='rsa3')) + \
            pn.geom_point(pn.aes(color='factor(size)'))
    pn.ggsave(p, 'vn_abx3_rsa3.pdf')
    res = [r for r in results if r['abx3'] is not None and r['rsa'] is not None]
    res = from_records(res)
    p = pn.ggplot(res, pn.aes(x='abx3', y='rsa')) + \
            pn.geom_point(pn.aes(color='factor(size)'))
    pn.ggsave(p, 'vn_abx3_rsa.pdf')
    return

#    # ABX
#    xrange = range_size
#    fig, ax = plt.subplots()
#    legend = []
#    for type in ['z', 'indices']:
#        for mode in ['random', 'trained']:
#            data = [100 - abx['english_triplets'][x][type][mode] for x in xrange]
#            ax.plot(range(0, len(xrange)), data)
#            legend.append('{} - {}'.format(type, mode))
#            ax.set(xlabel='Size of the codebook', ylabel='ABX accuracy')
#    ax.set_xticks(np.arange(len(xrange)))
#    ax.set_xticklabels(xrange)
#    ax.legend(legend)
#    ax.grid()
#    fig.savefig("vn_abx_triplets.pdf")
#    fig, ax = plt.subplots()
#    legend = []
#    for type in ['z', 'indices']:
#        for mode in ['random', 'trained']:
#            data = [100 - abx_within['english_triplets'][x][type][mode] for x in xrange]
#            ax.plot(range(0, len(xrange)), data)
#            legend.append('{} - {}'.format(type, mode))
#            ax.set(xlabel='Size of the codebook', ylabel='ABX accuracy')
#    ax.set_xticks(np.arange(len(xrange)))
#    ax.set_xticklabels(xrange)
#    ax.legend(legend)
#    ax.grid()
#    fig.savefig("vn_abx_within_triplets.pdf")
#
#    # RSA
#    fig, ax = plt.subplots()
#    legend = []
#    for mode in ['random', 'trained']:
#        for level in ['phoneme', 'word']:
#            data = [rsa['english_triplets'][x]['indices'][mode][level] for x in xrange]
#            ax.plot(range(0, len(xrange)), data)
#            legend.append('{} - {} - {}'.format(type, mode, level))
#            ax.set(xlabel='Size of the codebook', ylabel='RSA score')
#    ax.set_xticks(np.arange(len(xrange)))
#    ax.set_xticklabels(xrange)
#    ax.legend(legend)
#    ax.grid()
#    fig.savefig("vn_rsa_triplets.pdf")
#    fig, ax = plt.subplots()
#    legend = []
#    for mode in ['random', 'trained']:
#        for level in ['phoneme', 'word']:
#            data = [rsa['english'][x]['indices'][mode][level] for x in xrange]
#            ax.plot(range(0, len(xrange)), data)
#            legend.append('{} - {} - {}'.format(type, mode, level))
#            ax.set(xlabel='Size of the codebook', ylabel='RSA score')
#    ax.set_xticks(np.arange(len(xrange)))
#    ax.set_xticklabels(xrange)
#    ax.legend(legend)
#    ax.grid()
#    fig.savefig("vn_rsa.pdf")
#
#    # DC
#    fig, ax = plt.subplots()
#    legend = []
#    for mode in ['random', 'trained']:
#        for level in ['acc', 'baseline']:
#            data = [dc['english'][x]['indices'][mode][level] for x in xrange]
#            ax.plot(range(0, len(xrange)), data)
#            legend.append('{} - {} - {}'.format(type, mode, level))
#            ax.set(xlabel='Size of the codebook', ylabel='DC score')
#    ax.set_xticks(np.arange(len(xrange)))
#    ax.set_xticklabels(xrange)
#    ax.legend(legend)
#    ax.grid()
#    fig.savefig("vn_dc.pdf")
#
#    # DC vs. RSA
#    fig, ax = plt.subplots()
#    legend = []
#    for mode in ['trained']:
#        xdata = [dc['english'][x]['indices'][mode]['acc'] for x in xrange]
#        ydata = [rsa['english_triplets'][x]['indices'][mode]['phoneme'] for x in xrange]
#        for i in range(len(xdata)):
#            ax.plot(xdata[i], ydata[i], 'o')
#            legend.append('{}'.format(range_size[i]))
#        ax.set(xlabel='DC score', ylabel='RSA score')
#    ax.legend(legend)
#    ax.grid()
#    fig.savefig("vn_dc_rsa.pdf")


if __name__ == '__main__':
    plot_all()
