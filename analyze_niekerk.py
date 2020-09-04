import abx_trigrams as abx
from ABXpy.distance import default_distance
import logging
import os
from pathlib import Path
from prepare_niekerk import save_data, save_data_trigrams
from shutil import copyfile
from vq_eval import local_diag, rsa, rsa_trigrams

experiments = [('english/val', False), ('english_triplets/val', True)]
encroot = Path('/home/bjrhigy/dev/bshall-zrsc/submission/flickr8k')
outroot = Path('experiments/niekerk')

for exp, is_trigram in experiments:
    for encoding, distfun in [('z', default_distance), ('indices', abx.ed)]:
        for size in ['32', '64', '128', '256', '512', '1024']:
            for mode in ['trained', 'random']:
                srcdir = encroot / exp / size / encoding / mode
                encdir = srcdir / 'encodings'
                outdir = outroot / exp / size / encoding / mode
                logging.info("Analyzing {}".format(encdir))
                os.makedirs(outdir, exist_ok=True)
                # Computing ABX score
                # TODO adapt for full sentences
                if is_trigram:
                    logging.info("Computing ABX score")
                    for fname in ['features', 'distance', 'score', 'analyze']:
                        if os.path.isfile(encdir / fname):
                            os.remove(encdir / fname)
                    abx.compute_result(str(encdir),
                                       'data/flickr8k_abx_within.triplets',
                                       outdir, within_speaker=True,
                                       distancefun=distfun)
                # Preparing files for DC and RSA
                if encoding == 'indices' or not is_trigram:
                    logging.info("Processing files for DC and RSA scores")
                    outdir = outroot / exp / size / encoding
                    if is_trigram:
                        os.makedirs(outdir / 'trigrams', exist_ok=True)
                        copyfile(
                            outroot / 'downsampling_factors.json',
                            outdir / 'trigrams' / 'downsampling_factors.json')
                        save_data_trigrams(encdir, outdir / 'trigrams', mode)
                    else:
                        os.makedirs(outdir, exist_ok=True)
                        copyfile(outroot / 'downsampling_factors.json',
                                 outdir / 'downsampling_factors.json')
                        save_data(encdir, outdir, mode)
            if not is_trigram:
                logging.info("Computing DC score")
                local_diag([outdir])
            if encoding == 'indices':
                logging.info("Computing RSA score")
                if is_trigram:
                    rsa_trigrams([outdir])
                else:
                    rsa([outdir])
