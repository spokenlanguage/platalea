from extract_results import extract_abx, extract_rsa, extract_dc, range_size
import matplotlib.pyplot as plt
import numpy as np


abx = extract_abx()
rsa = extract_rsa()
dc = extract_dc()

# ABX
xrange = range_size
fig, ax = plt.subplots()
legend = []
for type in ['z', 'indices']:
    for mode in ['random', 'trained']:
        data = [100 - abx['english_triplets'][x][type][mode] for x in xrange]
        ax.plot(range(0, len(xrange)), data)
        legend.append('{} - {}'.format(type, mode))
        ax.set(xlabel='Size of the codebook', ylabel='ABX accuracy')
ax.set_xticks(np.arange(len(xrange)))
ax.set_xticklabels(xrange)
ax.legend(legend)
ax.grid()
fig.savefig("abx.pdf")

# RSA
fig, ax = plt.subplots()
legend = []
for mode in ['random', 'trained']:
    for level in ['phoneme', 'word']:
        data = [rsa['english_triplets'][x]['indices'][mode][level] for x in xrange]
        ax.plot(range(0, len(xrange)), data)
        legend.append('{} - {} - {}'.format(type, mode, level))
        ax.set(xlabel='Size of the codebook', ylabel='RSA score')
ax.set_xticks(np.arange(len(xrange)))
ax.set_xticklabels(xrange)
ax.legend(legend)
ax.grid()
fig.savefig("rsa_triplets.pdf")
fig, ax = plt.subplots()
legend = []
for mode in ['random', 'trained']:
    for level in ['phoneme', 'word']:
        data = [rsa['english'][x]['indices'][mode][level] for x in xrange]
        ax.plot(range(0, len(xrange)), data)
        legend.append('{} - {} - {}'.format(type, mode, level))
        ax.set(xlabel='Size of the codebook', ylabel='RSA score')
ax.set_xticks(np.arange(len(xrange)))
ax.set_xticklabels(xrange)
ax.legend(legend)
ax.grid()
fig.savefig("rsa.pdf")

# DC
fig, ax = plt.subplots()
legend = []
for mode in ['random', 'trained']:
    for level in ['acc', 'baseline']:
        data = [dc['english'][x]['indices'][mode][level] for x in xrange]
        ax.plot(range(0, len(xrange)), data)
        legend.append('{} - {} - {}'.format(type, mode, level))
        ax.set(xlabel='Size of the codebook', ylabel='DC score')
ax.set_xticks(np.arange(len(xrange)))
ax.set_xticklabels(xrange)
ax.legend(legend)
ax.grid()
fig.savefig("dc.pdf")

# ABX vs. RSA
fig, ax = plt.subplots()
legend = []
for mode in ['trained']:
    xdata = [100 - abx['english_triplets'][x]['indices'][mode] for x in xrange]
    ydata = [rsa['english_triplets'][x]['indices'][mode]['phoneme'] for x in xrange]
    for i in range(len(xdata)):
        ax.plot(xdata[i], ydata[i], 'o')
        legend.append('{}'.format(range_size[i]))
    ax.set(xlabel='ABX accuracy', ylabel='RSA score')
ax.legend(legend)
ax.grid()
fig.savefig("abx_rsa.pdf")

# DC vs. RSA
fig, ax = plt.subplots()
legend = []
for mode in ['trained']:
    xdata = [dc['english'][x]['indices'][mode]['acc'] for x in xrange]
    ydata = [rsa['english_triplets'][x]['indices'][mode]['phoneme'] for x in xrange]
    for i in range(len(xdata)):
        ax.plot(xdata[i], ydata[i], 'o')
        legend.append('{}'.format(range_size[i]))
    ax.set(xlabel='DC score', ylabel='RSA score')
ax.legend(legend)
ax.grid()
fig.savefig("dc_rsa.pdf")
