SEED=666
import torch
torch.manual_seed(SEED)
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)


import platalea.analysis.phoneme as P

config = dict(directory = '../../../data/out/trans/',
              size = 485195 // 2,
              layers=['convout'] + [ 'transf{}'.format(i) for i in range(12) ],
              matrix=False
              )

P.local_rsa(config)
P.local_rsa_plot()
