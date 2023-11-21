import json
import pickle

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def write_json_file(file_path, out):
    with open(file_path, 'w') as f:
        json.dump(out, f, indent='\t')  

def write_pickle_file(file_path, out):
    with open(file_path, 'wb') as f:
        pickle.dump(out, f)

id2text = read_json_file("../../data/original_version.json")
locid2title = read_json_file("../../data/metadata/localDocId2title.json")
results_q2id = {}
results_q2title = {}
results_q2locid = {}
results_locid2title = {}

for idx, text in id2text.items():
    results_q2id[text["qText"]] = text["qId"]
    results_q2title[text["qText"]] = []
    results_q2locid[text["qText"]] = text["localIdList"]
    for locid in text["localIdList"]:
        tit = locid2title[locid].split("Title: ")[1].split("[http")[0].strip()
        results_q2title[text["qText"]].append(tit)

for idx, text in locid2title.items():
    results_locid2title[idx] = text.split("Title: ")[1].split("[https:")[0].strip()

write_pickle_file("./qtext2qid.pickle", results_q2id)
write_pickle_file("./qtext2title.pickle", results_q2title)
write_pickle_file("./qtext2locid.pickle", results_q2locid)
write_pickle_file("./locid2title.pickle", results_locid2title)
