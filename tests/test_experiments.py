import unittest.mock
from flickr1d import __path__ as flickr1d_path
import deepdiff

flickr1d_path = flickr1d_path[-1]


def test_config():
    with unittest.mock.patch('sys.argv', ['[this gets ignored]', '--epochs=2', '--flickr8k_meta=thisandthat.json',
                                          '--audio_features_fn=mfcc_features.pt',
                                          f'--flickr8k_root={flickr1d_path}',
                                          '--lr_scheduler=noam', '-v']):
        from platalea.experiments.config import get_argument_parser

        args = get_argument_parser()
        args.parse()

        assert args.epochs == 2
        assert args.flickr8k_meta == 'thisandthat.json'
        assert args.verbose
        assert args.lr_scheduler == 'noam'


def test_transformer_experiment():
    expected = [{'epoch': 1,
                 'medr': 1.5,
                 'recall': {1: 0.5, 5: 1.0, 10: 1.0},
                 'average_loss': 0.5192822813987732,
                 }]

    with unittest.mock.patch('sys.argv', ['[this gets ignored]',
                                          '--epochs=1',
                                          '-c', f'{flickr1d_path}/config.yml',
                                          f'--flickr8k_root={flickr1d_path}',
                                          '--trafo_heads=4',
                                          '--trafo_d_model=4',
                                          '--trafo_feedforward_dim=4',
                                          '--trafo_dropout=0.1',
                                          '--loss_logging_interval=1',
                                          '--validation_interval=1',
                                          '--lr_scheduler=noam'
                                          ]):
        import platalea.experiments.flickr8k.transformer
        result = platalea.experiments.flickr8k.transformer.result

    _assert_nested_almost_equal(result, expected)


def test_basic_experiment():
    expected = [{'epoch': 1,
                 'medr': 1.5,
                 'recall': {1: 0.5, 5: 1.0, 10: 1.0},
                 'average_loss': 0.41894787549972534
                 }]

    with unittest.mock.patch('sys.argv', ['[this gets ignored]',
                                          '--epochs=1',
                                          '-c', f'{flickr1d_path}/config.yml',
                                          f'--flickr8k_root={flickr1d_path}',
                                          '--hidden_size_factor=4',
                                          '--optimizer=adadelta'
                                          ]):
        import platalea.experiments.flickr8k.basic
        result = platalea.experiments.flickr8k.basic.result

    _assert_nested_almost_equal(result, expected)


def test_mtl_asr_experiment():
    expected = [
        {'ASR': {'cer': {'CER': 6.791171477079796,
                         'Cor': 0,
                         'Del': 0,
                         'Ins': 3411,
                         'Sub': 589},
                 'average_loss': 4.440168142318726,
                 'wer': {'Cor': 0,
                         'Del': 118,
                         'Ins': 0,
                         'Sub': 10,
                         'WER': 1.0}},
         'SI': {'medr': 1.5,
                'recall': {1: 0.5,
                           5: 1.0,
                           10: 1.0},
                'average_loss': 0.3971380218863487},
         'epoch': 1}
    ]

    with unittest.mock.patch('sys.argv', ['[this gets ignored]',
                                          '--epochs=1',
                                          '-c', f'{flickr1d_path}/config.yml',
                                          f'--flickr8k_root={flickr1d_path}',
                                          '--hidden_size_factor=4',
                                          '--loss_logging_interval=1',
                                          '--validation_interval=1'
                                          ]):
        import platalea.experiments.flickr8k.mtl_asr
        result = platalea.experiments.flickr8k.mtl_asr.result

    _assert_nested_almost_equal(result, expected)


def test_mtl_st_experiment():
    expected = [
        {'SI': {'medr': 2.0, 'recall': {1: 0.4, 5: 1.0, 10: 1.0}, 'average_loss': 0.3906550034880638},
         'ST': {'medr': 6.0, 'recall': {1: 0.0, 5: 0.5, 10: 1.0}, 'average_loss': 0.37090546637773514},
         'epoch': 1},
    ]

    with unittest.mock.patch('sys.argv', ['[this gets ignored]',
                                          '--epochs=1',
                                          '-c', f'{flickr1d_path}/config.yml',
                                          f'--flickr8k_root={flickr1d_path}',
                                          '--hidden_size_factor=4',
                                          '--lr_scheduler=constant'
                                          ]):
        import platalea.experiments.flickr8k.mtl_st
        result = platalea.experiments.flickr8k.mtl_st.result

    _assert_nested_almost_equal(result, expected)


def test_asr_experiment():
    expected = [
        {'cer': {'CER': 6.784380305602716,
                 'Cor': 4,
                 'Del': 0,
                 'Ins': 3411,
                 'Sub': 585},
         'epoch': 1,
         'average_loss': 4.3757164478302,
         'wer': {'Cor': 0,
                 'Del': 118,
                 'Ins': 0,
                 'Sub': 10,
                 'WER': 1.0}},
    ]

    with unittest.mock.patch('sys.argv', ['[this gets ignored]',
                                          '--epochs=1',
                                          '-c', f'{flickr1d_path}/config.yml',
                                          f'--flickr8k_root={flickr1d_path}',
                                          '--hidden_size_factor=4',
                                          '--epsilon_decay=0.001',
                                          '--loss_logging_interval=1',
                                          '--validation_interval=1'
                                          ]):
        import platalea.experiments.flickr8k.asr
        result = platalea.experiments.flickr8k.asr.result

    _assert_nested_almost_equal(result, expected)


def test_text_image_experiment():
    expected = [{
        'epoch': 1,
        'medr': 1.5,
        'recall': {1: 0.5, 5: 1.0, 10: 1.0},
        'average_loss': 0.3847378194332123,
    }]

    with unittest.mock.patch('sys.argv', ['[this gets ignored]',
                                          '--epochs=1',
                                          '-c', f'{flickr1d_path}/config.yml',
                                          f'--flickr8k_root={flickr1d_path}',
                                          '--hidden_size_factor=4',
                                          '--loss_logging_interval=1',
                                          '--validation_interval=1'
                                          ]):
        import platalea.experiments.flickr8k.text_image
        result = platalea.experiments.flickr8k.text_image.result

    _assert_nested_almost_equal(result, expected)


def test_pip_ind_experiment():
    expected = {
        'ranks': [2, 2, 2, 2, 2, 1, 1, 1, 1, 1],
        'recall': {
            1: [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            5: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            10: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}
    }

    with unittest.mock.patch('sys.argv', ['[this gets ignored]',
                                          '--epochs=1',
                                          '-c', f'{flickr1d_path}/config.yml',
                                          f'--flickr8k_root={flickr1d_path}',
                                          '--hidden_size_factor=4',
                                          #   '--asr_model_dir={asr_out_path}'
                                          #   '--text_image_model_dir={text_image_out_path}'
                                          ]):
        import platalea.experiments.flickr8k.pip_ind
        result = platalea.experiments.flickr8k.pip_ind.result

    _assert_nested_almost_equal(result, expected)


def test_pip_seq_experiment():
    expected = [{'medr': 1.5, 'recall': {1: 0.5, 5: 1.0, 10: 1.0},
                 'average_loss': 0.36633214354515076,
                 'epoch': 1}]

    with unittest.mock.patch('sys.argv', ['[this gets ignored]',
                                          '--epochs=1',
                                          '-c', f'{flickr1d_path}/config.yml',
                                          f'--flickr8k_root={flickr1d_path}',
                                          '--hidden_size_factor=8',
                                          '--pip_seq_no_beam_decoding',
                                          '--flickr8k_language=jp',
                                          #   '--asr_model_dir={asr_out_path}'
                                          ]):
        import platalea.experiments.flickr8k.pip_seq
        result = platalea.experiments.flickr8k.pip_seq.result

    _assert_nested_almost_equal(result, expected)


def _assert_nested_almost_equal(a, b):
    """
    Asserts that 2 nested objects are approximately equal.
    """
    assert(not deepdiff.DeepDiff(a, b, math_epsilon=1e-5, ignore_numeric_type_changes=True))
