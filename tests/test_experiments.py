import unittest.mock
from flickr1d import __path__ as flickr1d_path

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
        assert platalea.experiments.flickr8k.transformer.result == {'final_loss': 0.5153712034225464}


def test_basic_experiment():
    with unittest.mock.patch('sys.argv', ['[this gets ignored]',
                                          '--epochs=1',
                                          '-c', f'{flickr1d_path}/config.yml',
                                          f'--flickr8k_root={flickr1d_path}',
                                          '--hidden_size_factor=4']):

        import platalea.experiments.flickr8k.basic
        assert platalea.experiments.flickr8k.basic.result == {'final_loss': 0.41894787549972534}


def test_mtl_asr_experiment():
    with unittest.mock.patch('sys.argv', ['[this gets ignored]',
                                          '--epochs=1',
                                          '-c', f'{flickr1d_path}/config.yml',
                                          f'--flickr8k_root={flickr1d_path}',
                                          '--hidden_size_factor=4']):
        import platalea.experiments.flickr8k.mtl_asr
        assert platalea.experiments.flickr8k.mtl_asr.result == {
            'ASR': {'cer': {'CER': 6.791171477079796,
                            'Cor': 0,
                            'Del': 0,
                            'Ins': 3411,
                            'Sub': 589},
                    'wer': {'Cor': 0,
                            'Del': 118,
                            'Ins': 0,
                            'Sub': 10,
                            'WER': 1.0}},
            'SI': {'medr': 1.5,
                   'recall': {1: 0.5,
                              5: 1.0,
                              10: 1.0}},
            'epoch': 1,
        }


def test_mtl_st_experiment():
    with unittest.mock.patch('sys.argv', ['[this gets ignored]',
                                          '--epochs=1',
                                          '-c', f'{flickr1d_path}/config.yml',
                                          f'--flickr8k_root={flickr1d_path}',
                                          '--hidden_size_factor=4']):
        import platalea.experiments.flickr8k.mtl_st
        assert platalea.experiments.flickr8k.mtl_st.result == {'SI': {'medr': 2.0, 'recall': {1: 0.4, 5: 1.0, 10: 1.0}},
                                                               'ST': {'medr': 6.0, 'recall': {1: 0.0, 5: 0.5, 10: 1.0}},
                                                               'epoch': 1}


def test_asr_experiment():
    with unittest.mock.patch('sys.argv', ['[this gets ignored]',
                                          '--epochs=1',
                                          '-c', f'{flickr1d_path}/config.yml',
                                          f'--flickr8k_root={flickr1d_path}',
                                          '--hidden_size_factor=4',
                                          '--epsilon_decay=0.001']):
        import platalea.experiments.flickr8k.asr
        assert platalea.experiments.flickr8k.asr.result == {'validation loss': 4.364380836486816}
    # save output of this experiment to serve as input for pip_ind and pip_seq


def test_text_image_experiment():
    with unittest.mock.patch('sys.argv', ['[this gets ignored]',
                                          '--epochs=1',
                                          '-c', f'{flickr1d_path}/config.yml',
                                          f'--flickr8k_root={flickr1d_path}',
                                          '--hidden_size_factor=4']):
        import platalea.experiments.flickr8k.text_image
        assert platalea.experiments.flickr8k.text_image.result == {
            'epoch': 1,
            'medr': 1.5,
            'recall': {1: 0.5, 5: 1.0, 10: 1.0}}
    # save output of this experiment to serve as input for pip_ind


def test_pip_ind_experiment():
    with unittest.mock.patch('sys.argv', ['[this gets ignored]',
                                          '--epochs=1',
                                          '-c', f'{flickr1d_path}/config.yml',
                                          f'--flickr8k_root={flickr1d_path}',
                                          '--hidden_size_factor=4',
                                        #   '--asr_model_dir={asr_out_path}'
                                        #   '--text_image_model_dir={text_image_out_path}'
                                          ]):
        import platalea.experiments.flickr8k.pip_ind
        assert platalea.experiments.flickr8k.pip_ind.result == {
            'ranks': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            'recall': {
                1: [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                5: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                10: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}}


def test_pip_seq_experiment():
    with unittest.mock.patch('sys.argv', ['[this gets ignored]',
                                          '--epochs=1',
                                          '-c', f'{flickr1d_path}/config.yml',
                                          f'--flickr8k_root={flickr1d_path}',
                                          '--hidden_size_factor=4',
                                          '--pip_seq_no_beam_decoding',
                                        #   '--asr_model_dir={asr_out_path}'
                                          ]):
        import platalea.experiments.flickr8k.pip_seq
        assert platalea.experiments.flickr8k.pip_seq.result == {'epoch': 1, 'medr': 1.5,
                                                                'recall': {1: 0.5, 5: 1.0, 10: 1.0}}
