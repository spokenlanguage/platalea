import json

M = json.load(open('dataset_multilingual_human.json'))


def filter(M):
    I = []
    for i, m in enumerate(M['images']):
        S = []
        for j, s in enumerate(m['sentences']):
            if 'raw_jp' in s:
                S.append(s)
            else:
                S.append(None)
        m['sentences'] = S
        if len(m['sentences']) != 0:
            I.append(m)
    M['images'] = I


def count_sent(M):
    cntr = 0
    for m in M['images']:
        for s in m['sentences']:
            cntr += 1
    return cntr


filter(M)
count_sent(M)
json.dump(M, open('dataset_multilingual_human_only.json', 'w'))
