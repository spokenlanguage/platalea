import argparse
import json
import torch

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('num_samples', help='Number of samples to keep.', type=int)
args = parser.parse_args()

meta = json.load(open('metadata.json'))
data = torch.load('mfcc_delta_features.pt')
# Split in subsets
indexes = {}
for subset in ['train', 'dev', 'test']:
    indexes[subset] = [i for i, m in enumerate(meta) if m['split'] == subset]
# Select examples
num_ex_dev_test = args.num_samples // 4
num_ex_train = args.num_samples - 2 * num_ex_dev_test
new_meta = [meta[i] for i in indexes['train'][0:num_ex_train]] + \
           [meta[i] for i in indexes['dev'][0:num_ex_dev_test]] + \
           [meta[i] for i in indexes['test'][0:num_ex_dev_test]]
new_filenames = [data['filenames'][i] for i in indexes['train'][0:num_ex_train]] + \
                [data['filenames'][i] for i in indexes['dev'][0:num_ex_dev_test]] + \
                [data['filenames'][i] for i in indexes['test'][0:num_ex_dev_test]]
new_features = [data['features'][i] for i in indexes['train'][0:num_ex_train]] + \
               [data['features'][i] for i in indexes['dev'][0:num_ex_dev_test]] + \
               [data['features'][i] for i in indexes['test'][0:num_ex_dev_test]]
new_data = {'filenames': new_filenames, 'features': new_features}
# Saving data
json.dump(new_meta, open('metadata_{}.json'.format(args.num_samples), 'w'))
torch.save(new_data, 'mfcc_delta_features_{}.pt'.format(args.num_samples))
