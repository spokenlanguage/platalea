import platalea.analysis.phoneme as P

config = dict(datadir = '../../../data/out/trans/',
              hidden=None,
              epochs=40,
              layers=['conv2', 'convout'] + [ 'transf{}'.format(i) for i in range(12) ]
              )

P.local_diagnostic(config)
P.local_diagnostic_plot()
