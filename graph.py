import os
import re
import pickle

from collections import defaultdict

from tqdm import tqdm

file_path = 'NLPCC2017-OpenDomainQA/knowledge/nlpcc-iccpol-2016.kbqa.kb'

graph = defaultdict(list)
entity_linking = defaultdict(list)
with open(file_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        s, p, o = line.strip().split(' ||| ')
        s = s.lower()
        p = p.replace(' ', '')
        o = o.lower()
        graph[s].append((s, p, o))
        if '(' in s:
            s1 = s.split('(')[0]
            entity_linking[s1].append(s)
        if s[0] =='《' and s[-1] == '》':
            entity_linking[s[1:-1]].append(s)
        
print('Dumping gragh...')
with open('graph/graph.pkl', 'wb') as f:
    pickle.dump(graph, f)

print('Dumping entity linking...')
with open('graph/entity_linking.pkl', 'wb') as f:
    pickle.dump(entity_linking, f)