SEED=666
import torch
torch.manual_seed(SEED)
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)

import platalea.analysis.phoneme as P

config = dict(directory = '../../../data/out/vgs-asr/',
              attention = 'mean',
              hidden_size = None,
              attention_hidden_size = None,
              epochs=500,
              test_size = 1/2,
              layers=['conv'] + [ 'rnn{}'.format(i) for i in range(4) ],
              device='cuda:2'
              )

P.global_diagnostic(config)
P.global_diagnostic_plot()
