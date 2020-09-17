import unittest.mock


def test_transformer_experiment():
    with unittest.mock.patch('sys.argv', ['--epochs=2', '--flickr8k_meta=dataset.json',
                                          '--audio_features_fn=mfcc_features.pt',
                                          '--flickr8k_root=/Users/pbos/projects/spokenLanguage/flickr1d_git/flickr1d',
                                          '--lr_scheduler=noam', '-v']):
        import platalea.experiments.flickr8k.transformer
