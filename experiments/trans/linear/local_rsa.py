import random
random.seed(123)

import platalea.analysis.phoneme as P

config = dict(datadir = '../../../data/out/trans/',
              #size = 20000,
              size = 485195 // 2,
              layers=['conv2', 'convout'] + [ 'transf{}'.format(i) for i in range(12) ],
              matrix=False
              )

P.local_rsa(config)
P.local_rsa_plot()
