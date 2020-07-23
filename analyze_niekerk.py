import abx_trigrams as abx
from ABXpy.distance import default_distance
import os
from pathlib import Path
from prepare_niekerk import save_data, save_data_trigrams
from shutil import copyfile
from vq_eval import local_diag, rsa

experiments = ['english/val', 'english_triplets/val']
encroot = Path('/home/bjrhigy/dev/bshall-zrsc/submission/flickr8k')
outroot = Path('experiments/niekerk')

experiments = ['english/val']

#for exp in experiments:
#    for encoding, distfun in [('z', default_distance), ('indices', abx.ed)]:
#        for mode in ['trained', 'random']:
#            srcdir = encroot / exp / encoding / mode
#            encdir = srcdir / 'encodings'
#            outdir = outroot / exp / encoding / mode
#            os.makedirs(outdir, exist_ok=True)
##            # Computing ABX score
##            for fname in ['features', 'distance', 'score']:
##                if os.path.isfile(srcdir / fname):
##                    os.remove(srcdir / fname)
##            abx.compute_result(encdir, 'data/flickr8k_abx.triplets', outdir,
##                               distancefun=distfun)
#            # Computing DC and RSA scores
#            outdir = outroot / exp / encoding
#            copyfile(outroot / 'downsampling_factors.json',
#                     outdir / 'downsampling_factors.json')
#            save_data(encdir, outdir, mode)
#        local_diag([outdir])
#        rsa([outdir])

experiments = ['english_triplets/val']

for exp in experiments:
    for encoding, distfun in [('z', default_distance), ('indices', abx.ed)]:
        for mode in ['trained', 'random']:
            srcdir = encroot / exp / encoding / mode
            encdir = srcdir / 'encodings'
            outdir = outroot / exp / encoding / mode
            os.makedirs(outdir, exist_ok=True)
            # Computing ABX score
            for fname in ['features', 'distance', 'score']:
                if os.path.isfile(srcdir / fname):
                    os.remove(srcdir / fname)
            abx.compute_result(encdir, 'data/flickr8k_abx.triplets', outdir,
                               distancefun=distfun)
            # Computing RSA scores
            outdir = outroot / exp / encoding
            copyfile(outroot / 'downsampling_factors.json',
                     outdir / 'downsampling_factors.json')
            save_data_trigrams(encdir, outdir)
        rsa_trigrams([outdir])
