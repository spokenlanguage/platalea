import argparse
from pathlib import Path
import json
import numpy as np
import torch
import platalea.basicvq
from tqdm import tqdm


def encode_dataset(args, params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model.load_state_dict(checkpoint["model"])
    model = torch.load(args.model).to(device)
    model.eval()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    hop_length_seconds = params["preprocessing"]["hop_length"] / params["preprocessing"]["sample_rate"]

    in_dir = Path(args.in_dir)
    for path in tqdm(in_dir.rglob("*.mel.npy")):
        mel = torch.from_numpy(np.load(path)).unsqueeze(0).to(device)
        with torch.no_grad():
            x = model.encoder(mel)
            x, _, _ = model.codebook(x)
        codes = ((model.codebook.embedding - x.squeeze().unsqueeze(dim=1))**2).sum(dim=2).argmin(dim=1)
        codes = torch.nn.functional.one_hot(codes, num_classes=model.codebook.embedding.shape[0]).cpu().numpy().astype(int)
        if args.discrete:
            output = codes
        else:
            output = x.squeeze().cpu().numpy()
        time = np.linspace(0, (mel.size(-1) - 1) * hop_length_seconds, len(output))
        relative_path = path.relative_to(in_dir).with_suffix("")
        out_path = out_dir / relative_path
        out_path.parent.mkdir(exist_ok=True, parents=True)
        
        if args.format == "npz":
            np.savez(out_path.with_suffix(".npz"), features=output, time=time)
        elif args.format == "txt":
            np.savetxt(out_path.with_suffix(".txt"), output, fmt="%d" if args.discrete else "%f")
        else:
            raise ValueError("Unknown format string")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--checkpoint", type=str, help="Checkpoint path to resume")
#     parser.add_argument("--in-dir", type=str, help="Directory to encode")
#     parser.add_argument("--out-dir", type=str, help="Output path")
#     parser.add_argument("--format", type=str, help="Output format")
#     parser.add_argument("--discrete", action='store_true', help="Use code IDs instead of embeddings")
#     args = parser.parse_args()
#     with open("config.json") as file:
#         params = json.load(file)
#     encode_dataset(args, params)

import glob
import torch
from platalea.preprocessing import audio_features
paths = glob.glob("sampleaudio/*.wav")
config = dict(type='mfcc', delta=True, alpha=0.97, n_filters=40,  window_size=0.025, frame_shift=0.010)
res = audio_features(paths, config)[1:]
model = torch.load("experiments/vq-256-q1/net.32.pt")

import torch
torch.manual_seed(123)
import logging
import platalea.basicvq as M
import platalea.dataset as D

logging.basicConfig(level=logging.INFO)

logging.info('Loading data')
data = dict(train=D.flickr8k_loader(split='train', batch_size=32, shuffle=True),
            val=D.flickr8k_loader(split='val', batch_size=32, shuffle=False))
D.Flickr8KData.init_vocabulary(data['train'].dataset)

bidi = True
config = dict(SpeechEncoder=dict(SpeechEncoderBottom=dict(conv=dict(in_channels=39, out_channels=64, kernel_size=6, stride=2, padding=0, bias=False),
                                                          rnn= dict(input_size=64, hidden_size=1024, num_layers=1,
                                                                    bidirectional=bidi, dropout=0)),
                                 VQEmbedding=dict(num_codebook_embeddings=256, embedding_dim=2 * 1024 if bidi else 1024, jitter=0.12),
                                 SpeechEncoderTop=dict(rnn= dict(input_size=2 * 1024 if bidi else 1024, hidden_size=1024, num_layers=3,
                                                                 bidirectional=bidi, dropout=0),
                                                       att= dict(in_size=2048, hidden_size=128))),
              ImageEncoder=dict(linear=dict(in_size=2048, out_size=2*1024), norm=True),
              margin_size=0.2)


logging.info('Building model')
net = M.SpeechImage(config)
net.load_state_dict(model.state_dict())
net.cuda()

codes = net.code_audio(res)

print(len(res))
for r in res:
    print("r", r.shape)
    print()
for code in codes:
    print("c", code)




