SEED=666
import torch
torch.manual_seed(SEED)
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)


import platalea.analysis.phoneme as P

config = dict(directory = '../../../data/out/vgs/',
              hidden=None,
              epochs=40,
              layers=['conv'] + [ 'rnn{}'.format(i) for i in range(4) ]
              )

P.local_diagnostic(config)
P.local_diagnostic_plot()
