SEED=666
import torch
torch.manual_seed(SEED)
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)

import platalea.analysis.phoneme as P

config = dict(directory = '../../../data/out/vgs/',
              attention = 'mean',
              epochs = 60,
              test_size = 1/2,
              layers=['conv'] + [ 'rnn{}'.format(i) for i in range(4) ],
              device = 'cpu'
              )

P.global_rsa(config)
P.global_rsa_plot()

