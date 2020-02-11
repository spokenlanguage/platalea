import os

import torch
import torch.utils.data
import json

class Flickr8KData(torch.utils.data.Dataset):

    def __init__(self, root, split='train'):
        self.root = root
        self.split = split
        self.metadata = json.load(open(root + 'dataset.json'))['images']
        # mapping from image id to list of caption id
        self.image_captions = {}
        for line in open(root + 'wav2capt.txt'):
            audio_id, image_id, text_id = line.split()
            text_id = int(text_id[1:])
            self.image_captions[image_id] = self.image_captions.get(image_id, []) + [(text_id, audio_id)]

        # image, caption pairs
        self.split_data = []
        for image in self.metadata:
            if image['split'] == self.split:
                for text_id, audio_id in self.image_captions[image['filename']]:
                    self.split_data.append((image['filename'], audio_id, image['sentences'][text_id]['raw']))

        # image and audio feature data
        image = torch.load(root + 'resnet_features.pt')
        self.image = dict(zip(image['filenames'], image['features']))
        audio = torch.load(root + 'mfcc_features.pt')
        self.audio = dict(zip(audio['filenames'], audio['features']))
        
    def __getitem__(self, index):
        image = self.image[self.split_data[index][0]]
        audio = self.audio[self.split_data[index][1]]
        text  = torch.Tensor([ord('^')] + [ ord(c) for c in self.split_data[index][2] ] + [ord('$')] ) # FIXME need to do this properly
        return dict(image_id=self.split_data[index][0],
                    audio_id=self.split_data[index][1],
                    image=image,
                    text=text,
                    audio=audio,
                    gloss= self.split_data[index][2])
                           
    def __len__(self):
        return len(self.split_data)

    def evaluation(self):
        """Returns image features, caption features, and a boolean array specifying whether a caption goes with an image."""
        audio = []
        text  = []
        image = []
        matches = []
        for img in self.metadata:
            if img['split'] == self.split:
                image.append(self.image[img['filename']])
                for text_id, audio_id in self.image_captions[img['filename']]:
                    text.append(img['sentences'][text_id])
                    audio.append(self.audio[audio_id])
                    matches.append((len(audio)-1, len(image)-1))
        correct = torch.zeros(len(audio), len(image)).bool()
        for i,j in matches:
            correct[i,j] = True
        return dict(image=image, audio=audio, text=text, correct=correct)
    
def batch_audio(audios, max_frames=2048):
    """Merge audio captions. Truncate to max_frames. Pad with zeros."""
    mfcc_lengths = [len(cap[:max_frames,:]) for cap in audios]
    mfcc = torch.zeros(len(audios), max(mfcc_lengths), audios[0].size(1)) 
    for i, cap in enumerate(audios):
        end = mfcc_lengths[i]
        mfcc[i, :end] = cap[:end]
    return mfcc.permute(0, 2, 1), torch.tensor(mfcc_lengths)

def batch_text(texts):
    """Merge captions (from tuple of 1D tensor to 2D tensor). Pad with zeros."""
    # FIXME this needs to be done properly, eventually
    char_lengths = [len(cap) for cap in texts]
    chars = torch.zeros(len(texts), max(char_lengths)).long()
    for i, cap in enumerate(texts):
        end = char_lengths[i]
        chars[i, :end] = cap[:end]        
    return chars, torch.tensor(char_lengths)

def batch_image(images):
    return torch.stack(images, 0)

def collate_fn(data, max_frames=2048):
    #data.sort(key=lambda x: len(x[1]), reverse=True)
    images, texts, audios = zip(* [ (datum['image'], datum['text'], datum['audio']) for datum in data ])

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = batch_image(images)

    mfcc, mfcc_lengths = batch_audio(audios, max_frames=max_frames)
    chars, char_lengths = batch_text(texts)
    
    return dict(image=images, audio=mfcc, text=chars, audio_len=mfcc_lengths, text_len=char_lengths)

def flickr8k_loader(split='train', batch_size=32, shuffle=False, max_frames=2048):
    data_root = os.environ["PLATALEA_DATA_ROOT"] + '/'
    return torch.utils.data.DataLoader(dataset=Flickr8KData(root=data_root, split=split), 
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=0,
                                       collate_fn=lambda x: collate_fn(x, max_frames=max_frames))

        
