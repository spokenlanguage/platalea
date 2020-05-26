import platalea.analysis.phoneme as P

config = dict(directory = '../../../data/out/trans/',
              attention = 'mean',
              hidden_size = None,
              attention_hidden_size = None,
              epochs=500,
              test_size = 1/2,
              layers=['convout'] + [ 'transf{}'.format(i) for i in range(12) ],
              device='cuda:1'
              )

P.global_diagnostic(config)
P.global_diagnostic_plot()
