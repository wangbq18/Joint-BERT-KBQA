import os
import json

from collections import defaultdict
from copy import deepcopy

from tqdm import tqdm

def load_data(file_path):
    data = []
    num = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        sample = {}
        for line in f:
            if line.startswith('<question'):
                sample['question'] = line.strip().split('\t')[1].strip().lower()
            elif line.startswith('<triple'):
                s, p, o = line.replace('\n', '').split('\t')[1].split(' ||| ')
                sample['triple'] = s.lower(), p.replace(' ', ''), o.lower()
                assert len(sample['triple']) == 3, (line, sample['triple'])
            elif line.startswith('==='):
                if sample['triple'][0] in sample['question']:
                    data.append(sample)
                else:
                    num += 1
                sample = {}
            else:
                pass
    print('num of failed sample: {}'.format(num))
    print('num of training sample: {}'.format(len(data)))
    return data

def load_knowledge(file_path='NLPCC2017-OpenDomainQA/knowledge/nlpcc-iccpol-2016.kbqa.kb'):
    s2p = defaultdict(list)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            s, p, _ = [tmp.strip() for tmp in line.strip().split(' ||| ')]
            s = s.lower()
            p = p.replace(' ', '')
            s2p[s].append(p)
    return s2p

if __name__ == "__main__":
    train_data = load_data('data/nlpcc-iccpol-2016.kbqa.training-data')
    s2p = load_knowledge()
    for item in tqdm(train_data):
        s, p, _ = item['triple']
        ps = deepcopy(s2p[s])
        if p in ps:
            ps.remove(p)
        if len(ps) > 0:
            item['negative_predicates'] = ps
        else:
            for ss in s2p:
                if s in ss and p in s2p[ss]:
                    ps = deepcopy(s2p[ss])
                    ps.remove(p)
                    item['negative_predicates'] = ps
                    break

    with open('data/train.json', 'w', encoding='utf-8') as f:
        for sample in train_data:
            f.write(json.dumps(sample, ensure_ascii=False)+'\n')

    test_data = load_data('data/nlpcc-iccpol-2016.kbqa.testing-data')
    with open('data/dev.json', 'w', encoding='utf-8') as f:
        for sample in test_data:
            f.write(json.dumps(sample, ensure_ascii=False)+'\n')


