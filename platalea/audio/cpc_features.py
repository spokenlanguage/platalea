import argparse
import json
import os

from cpc.feature_loader import buildFeature, FeatureModule, loadModel


def read_args(path_args):
    print(f"Loading args from {path_args}")
    with open(path_args, 'r') as file:
        args = argparse.Namespace(**json.load(file))
    return args


def load_feature_maker_CPC(cp_path, gru_level=-1, on_gpu=True):
    assert cp_path[-3:] == ".pt"
    assert os.path.exists(cp_path), \
        f"CPC path at {cp_path} does not exist!!"

    pathConfig = os.path.join(os.path.dirname(cp_path), "checkpoint_args.json")
    CPC_args = read_args(pathConfig)

    # Load FeatureMaker
    if gru_level is not None and gru_level > 0:
        updateConfig = argparse.Namespace(nLevelsGRU=gru_level)
    else:
        updateConfig = None

    model = loadModel([cp_path], updateConfig=updateConfig)[0]

    feature_maker = FeatureModule(model, CPC_args.onEncoder)
    feature_maker.eval()
    if on_gpu:
        feature_maker.cuda()
    return feature_maker


def cpc_feature_extraction(feature_maker, x, seq_norm=False, strict=True,
                           max_size_seq=10240):
    return buildFeature(feature_maker, x,
                        strict=strict,
                        maxSizeSeq=max_size_seq,
                        seqNorm=seq_norm)
