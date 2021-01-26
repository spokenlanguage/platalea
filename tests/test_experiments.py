import unittest.mock
from flickr1d import __path__ as flickr1d_path

flickr1d_path = flickr1d_path[-1]


def test_config():
    with unittest.mock.patch('sys.argv', ['[this gets ignored]', '--epochs=2', '--flickr8k_meta=thisandthat.json',
                                        '--audio_features_fn=mfcc_features.pt',
                                        f'--flickr8k_root={flickr1d_path}',
                                        '--lr_scheduler=noam', '-v']):
        from platalea.experiments.config import args

        args.parse()

        assert args.epochs == 2
        assert args.flickr8k_meta == 'thisandthat.json'
        assert args.verbose == True
        assert args.lr_scheduler == 'noam'


def test_transformer_experiment():
    with unittest.mock.patch('sys.argv', ['[this gets ignored]',
                                          '--epochs=1',
                                          '-c', f'{flickr1d_path}/config.yml',
                                          f'--flickr8k_root={flickr1d_path}',
                                          '--trafo_heads=4',
                                          '--trafo_d_model=4',
                                          '--trafo_feedforward_dim=4']):
        import platalea.experiments.flickr8k.transformer


def test_basic_default_experiment():
    with unittest.mock.patch('sys.argv', ['[this gets ignored]',
                                          '--epochs=1',
                                          '-c', f'{flickr1d_path}/config.yml',
                                          f'--flickr8k_root={flickr1d_path}',
                                          '--hidden_size_factor=4']):
        import platalea.experiments.flickr8k.basic_default


def test_mtl_asr_experiment():
    with unittest.mock.patch('sys.argv', ['[this gets ignored]',
                                          '--epochs=1',
                                          '-c', f'{flickr1d_path}/config.yml',
                                          f'--flickr8k_root={flickr1d_path}',
                                          '--hidden_size_factor=4']):
        import platalea.experiments.flickr8k.mtl_asr


def test_mtl_st_experiment():
    with unittest.mock.patch('sys.argv', ['[this gets ignored]',
                                          '--epochs=1',
                                          '-c', f'{flickr1d_path}/config.yml',
                                          f'--flickr8k_root={flickr1d_path}',
                                          '--hidden_size_factor=4']):
        import platalea.experiments.flickr8k.mtl_st
