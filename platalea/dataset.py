import torch
import torch.utils.data
import json

class Flickr8KData(torch.utils.data.Dataset):

    def __init__(self, root, split='train'):
        self.root = root
        self.split = split

        # mapping from image id to list of caption id
        self.image_captions = {}
        for line in open(root + 'wav2capt.txt'):
            audio_id, image_id, text_id = line.split()
            text_id = int(text_id[1:])
            self.image_captions[image_id] = self.image_captions.get(image_id, []) + [(text_id, audio_id)]

        # image, caption pairs
        self.split_data = []
        for image in json.load(open(root + 'dataset.json'))['images']:
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
        return image, text, audio
                           
    def __len__(self):
        return len(self.split_data)


def collate_fn(data, max_frames=2048):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, texts, audios = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions
    # truncate to max_frames
    # pad with zeros
    mfcc_lengths = [len(cap[:max_frames,:]) for cap in audios]
    mfcc = torch.zeros(len(audios), max(mfcc_lengths), audios[0].size(1)) 
    for i, cap in enumerate(audios):
        end = mfcc_lengths[i]
        mfcc[i, :end] = cap[:end]

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    # pad with zeros
    # FIXME this needs to be done properly, eventually
    char_lengths = [len(cap) for cap in texts]
    chars = torch.zeros(len(texts), max(char_lengths)).long()
    for i, cap in enumerate(texts):
        end = char_lengths[i]
        chars[i, :end] = cap[:end]        
    return dict(image=images, audio=mfcc, text=chars, audio_len=mfcc_lengths, text_len=char_lengths)

def flickr8k_loader(split='train', batch_size=32, shuffle=False, max_frames=2048):
    return torch.utils.data.DataLoader(dataset=Flickr8KData(root='/roaming/gchrupal/datasets/flickr8k/', split=split), 
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=0,
                                       collate_fn=lambda x: collate_fn(x, max_frames=max_frames))
