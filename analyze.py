import abx_trigrams as abx
from ABXpy.distance import default_distance
import os
from pathlib import Path
from prepare_niekerk import save_data, save_data_trigrams
from shutil import copyfile
from vq_eval import local_diag, rsa

encroot = Path('/home/bjrhigy/dev/bshall-zrsc/submission/flickr8k')
outroot = Path('experiments/niekerk')
experiments = ['english_triplets/val']

for exp in experiments:
    encoding = 'z'
    distfun = default_distance
    mode = 'trained'
    srcdir = encroot / exp
    encdir = srcdir
    outdir = outroot / exp / encoding / mode
    os.makedirs(outdir, exist_ok=True)
    # Computing ABX score
    abx.compute_result(encdir, 'data/flickr8k_abx.triplets', outdir)

