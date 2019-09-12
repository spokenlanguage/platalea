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
            caption_id, image_id, _ = line.split()
            self.image_captions[image_id] = self.image_captions.get(image_id, []) + [caption_id]

        # image, caption pairs
        self.split_data = []
        for image in json.load(open(root + 'dataset.json'))['images']:
            for caption_id in self.image_captions[image['filename']]:
                self.split_data.append((image['filename'], caption_id))

        # image and audio feature data
        image = torch.load(root + 'resnet_features.pt')
        self.image = dict(zip(image['filenames'], image['features']))
        audio = torch.load(root + 'mfcc_features.pt')
        self.audio = dict(zip(audio['filenames'], audio['features']))
        
    def __getitem__(self, index):
        image = self.image[self.split_data[index][0]]
        audio = self.audio[self.split_data[index][1]]
        return image, audio
                           
    def __len__(self):
        return len(self.split_data)


def collate_fn(data):
    max_frames = 2048
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    
    # Merge captions
    # truncate to max_frames
    # pad with zeros
    lengths = [len(cap[:max_frames,:]) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths), captions[0].size(1)) 
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths
