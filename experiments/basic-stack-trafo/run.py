import logging
import platalea.basic as M
import platalea.encoders
import platalea.dataset as D
import platalea.config

logging.basicConfig(level=logging.INFO)

logging.info('Loading data')
data = dict(train=D.flickr8k_loader(split='train', batch_size=32, shuffle=True),
            val=D.flickr8k_loader(split='val', batch_size=32, shuffle=False))
D.Flickr8KData.init_vocabulary(data['train'].dataset)

trafo_d_model = 512
speech_config = {'conv': dict(in_channels=39, out_channels=64, kernel_size=6, stride=2, padding=0, bias=False),
                 'trafo': dict(d_model=trafo_d_model, dim_feedforward=1024, num_encoder_layers=4, dropout=0, nhead=8),
                 'upsample': dict(mode='nearest'),
                 'att': dict(in_size=trafo_d_model, hidden_size=128),
                 }
speech_encoder = platalea.encoders.SpeechEncoderTransformer(speech_config)

# these must match, otherwise the loss cannot be calculated
image_encoder_out_size = trafo_d_model

config = dict(SpeechEncoder=speech_encoder,
              ImageEncoder=dict(linear=dict(in_size=2048, out_size=image_encoder_out_size), norm=True),
              margin_size=0.2)

logging.info('Building model')
net = M.SpeechImage(config)
run_config = dict(max_lr=2 * 1e-4, epochs=platalea.config.args.epochs)

logging.info('Training')
M.experiment(net, data, run_config)
