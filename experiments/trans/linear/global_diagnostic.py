import platalea.analysis.phoneme as P

config = dict(datadir = '../../../data/out/trans/',
              hidden=None,
              epochs=500,
              test_size = 1/2,
              #layers=['conv2', 'convout'] + [ 'transf{}'.format(i) for i in range(12) ]
              layers=['convout'] + [ 'transf{}'.format(i) for i in range(12) ]
              
              )

P.global_diagnostic(config)
P.global_diagnostic_plot()
