import platalea.analysis.phoneme as P

config = dict(directory = '../../../data/out/trans/',
              hidden=None,
              epochs=40,
              layers=['convout'] + [ 'transf{}'.format(i) for i in range(12) ]
              )

#P.local_diagnostic(config)
P.local_diagnostic_plot()
