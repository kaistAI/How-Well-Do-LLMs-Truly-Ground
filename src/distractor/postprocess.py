import json
import jsonlines
import pickle

def write_json_file(file_path, out):
    with open(file_path, 'w') as f:
        json.dump(out, f, indent='\t')  

def read_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def read_jsonl_file(file_path):
    data = []
    with jsonlines.open(file_path, 'r') as reader:
        for instance in reader:
            data.append(instance)
    return data

data = read_jsonl_file("./output/query_for_contriever.json")
qtext2title = read_pickle_file("./qtext2title.pickle")
qtext2locid = read_pickle_file("./qtext2locid.pickle")
qtext2qid = read_pickle_file("./qtext2qid.pickle")
locid2title = read_pickle_file("./locid2title.pickle")

n = 100 #top-n
results = {}
query = []
top_n = []
locid = []
gold_tit = []
pred_tit = []
for ins in data:
    num_candi=0
    qid = qtext2qid[ins['question']]
    q_title = set(qtext2title[ins['question']])
    dup = []
    idx = []
    for cxt in ins['ctxs']:
        if locid2title[cxt['id']] in q_title:
            continue
        if cxt['text'] in dup:
            continue
        idx.append(cxt['id'])
        locid.append(cxt['id'])
        query.append(ins['question'])
        gold_tit.append(q_title)
        pred_tit.append(locid2title[cxt['id']])
        top_n.append(cxt['text'])
        dup.append(cxt['text'])
        num_candi += 1
        if num_candi == n:
            break
    results[qid] = idx
write_json_file(f'output/distractor_qId2localId_top{n}.json', results)

import pandas as pd
df = pd.DataFrame({'query': query, f'top_{n}': top_n, 'loc_id' : locid, 'pred_title':pred_tit, 'gold_title': gold_tit})
df.to_csv(f'output/distractor_analysis_top{n}.csv', index=False)
