SEED=666
import torch
torch.manual_seed(SEED)
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)


import platalea.analysis.phoneme as P

config = dict(directory = '../../../data/out/trans/',
              attention = 'linear',
              hidden_size = None,
              attention_hidden_size = None,
              standardize = True,
              epochs=500,
              factor=0.5,
              test_size = 1/2,
              layers=['convout'] + [ 'transf{}'.format(i) for i in range(12) ],
              device='cuda:2'
              )

P.global_diagnostic(config)
P.global_diagnostic_plot()
