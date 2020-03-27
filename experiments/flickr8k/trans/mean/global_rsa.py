import platalea.analysis.phoneme as P

config = dict(directory = '../../../data/out/trans/',
              attention = 'mean',
              attention_hidden_size = None,
              epochs = 60,
              test_size = 1/2,
              layers=['convout'] + [ 'transf{}'.format(i) for i in range(12) ],
              device = 'cuda:0'
              )

P.global_rsa(config)
P.global_rsa_plot()
