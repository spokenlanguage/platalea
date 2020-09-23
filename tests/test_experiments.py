import unittest.mock


def test_config():
    with unittest.mock.patch('sys.argv', ['[this gets ignored]', '--epochs=2', '--flickr8k_meta=thisandthat.json',
                                        '--audio_features_fn=mfcc_features.pt',
                                        '--flickr8k_root=/Users/pbos/projects/spokenLanguage/flickr1d_git/flickr1d',
                                        '--lr_scheduler=noam', '-v']):
        from platalea.experiments.config import args

        args.parse()

        assert args.epochs == 2
        assert args.flickr8k_meta == 'thisandthat.json'
        assert args.verbose == True
        assert args.lr_scheduler == 'noam'


def test_transformer_experiment():
    with unittest.mock.patch('sys.argv', ['[this gets ignored]', '--epochs=2', '--flickr8k_meta=dataset.json',
                                          '--audio_features_fn=mfcc_features.pt',
                                          '--flickr8k_root=/Users/pbos/projects/spokenLanguage/flickr1d_git/flickr1d',
                                          '--lr_scheduler=noam', '-v']):
        import platalea.experiments.flickr8k.transformer
