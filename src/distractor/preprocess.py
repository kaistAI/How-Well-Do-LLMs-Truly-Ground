import json
import pandas as pd
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

id2text_input = read_json_file("../../data/original_version.json")
id2text_corpus = read_json_file("../../data/metadata/localDocId2localDocText.json")
id2title = read_json_file("../../data/metadata/localDocId2title.json")

# make corpus
Id = []
Text = []
Title = []
for idx, text in id2text_corpus.items():
    Id.append(idx)
    Text.append(text)
    Title.append(" ")
df = pd.DataFrame({'id' : Id, 'text' : Text, 'title' : Title})
df.to_csv("./corpus_for_contriever.tsv", sep='\t', index=False)

#make input query
results = []
for idx, text in id2text_input.items():
    results.append({'question': text['qText'], 'local_id': text['localIdList']})
write_json_file("./query_for_contriever.json", results)
