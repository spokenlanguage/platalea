SEED=666
import torch
torch.manual_seed(SEED)
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)


import platalea.analysis.phoneme as P

config = dict(directory = '../../../data/out/vgs/',
              size = 793964 // 2,
              layers=['conv'] + [ 'rnn{}'.format(i) for i in range(4) ],
              matrix=False
              )

P.local_rsa(config)
P.local_rsa_plot()
