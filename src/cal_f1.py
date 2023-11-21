import os
import re
import sys 
import nltk
import json
import torch
import spacy
import string
import argparse
import numpy as np
import torch.nn.functional as F

from typing import *
from tqdm import tqdm
from argparse import ArgumentParser
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def save_json(path, _dict):
    with open(path, "w") as f:
        json.dump(_dict, f)

def _check_type(elem):
    if type(elem) == list:
        assert len(elem) == 1
        return elem[0]
    else:
        assert type(elem) == str
        return elem

def get_pred_output(pred_dict):
    _input = _check_type(pred_dict["input"])
    _output = _check_type(pred_dict["output"])
    _output = _output.replace(_input, "")
    return _output

def cal_TRUE_score(_g_para, atomic_facts):
    _input = f"premise: {_g_para} hypothesis: {atomic_facts}"
    tok_ret = tokenizer(_input, return_tensors='pt', max_length=512, truncation=True)
    input_ids = tok_ret['input_ids'].cuda()
    attention_mask = tok_ret['attention_mask'].cuda()
    c_output = model.generate(input_ids, attention_mask=attention_mask, output_scores=True, return_dict_in_generate=True)
    c_score = tokenizer.batch_decode(
        c_output['sequences'],
        skip_special_tokens=True
    )[0] == "1"
    logits = F.softmax(c_output["scores"][0][0], dim=0)[209].detach().cpu().item() # ids of "1" is 209
    return c_score, logits

def cal_bi_score(atomic_facts, para, thres):
    atomic_emb = model.encode(atomic_facts)
    para_emb = model.encode([para])
    score = util.dot_score(atomic_emb, para_emb)
    values, indices = torch.topk(score, k=1)
    g_score = [int(el.item()>thres) for el in values]

def cal_cross_score(para, atomic_facts, thres):
    pred_list = [(atomic_fact, para) for atomic_fact in atomic_facts]
    scores = model.predict(pred_list)
    assert len(scores) == len(atomic_facts)
    score = [int(el>thres) for el in scores]
    return score

def get_TRUE_score(_g_para_list, atomic_fact):
    if type(_g_para_list) == list:
        # iterate over _g_para_list
        logit_list = []; score_list = []
        for _g_para in _g_para_list:
            _score, _logits = cal_TRUE_score(_g_para, atomic_fact)
            score_list.append(_score)
            logit_list.append(_logits)
        if True in score_list: 
            score = 1
        else:
            score = 0
        logit = np.array(logit_list).mean()
        return score, logit
    else:
        # assert len(_g_para_list) == 1
        return cal_TRUE_score(_g_para_list, atomic_fact) 

def get_gold_atomic_dict(annot_dict):
    qId2atomic = {}
    for qId, val_dict in annot_dict.items():
        atomic = val_dict["atomic_texts"]
        qId2atomic[qId] = list(set(atomic))
    return qId2atomic

def cleanse_sen(sen):
    sen = sen.replace('<s>', '').replace('</s>', '')
    return sen

def get_pred_atomic_dict(annot_dict):
    assert len(annot_dict.keys()) == 480
    qId2atomic = {}
    atomic_num = []
    for qId, val_list in annot_dict.items():
        atomic_list = []
        if type(val_list) == dict:
            val_list = val_list["atomic_facts"]
        for sen, _atomic_list in val_list:
            if len(val_list)!=1 and len(sen.split(" "))==1:
                # remove the single word case
                continue
            elif len(val_list)==1 and len(_atomic_list)==0:
                atomic_list.extend(sen) 
            else:
                sen_num = len(sen.split(" "))
                for _atomic in _atomic_list:
                    if sen_num < len(_atomic.split(' ')):
                        atomic_list.append(sen) 
                    else:
                        atomic_list.append(_atomic)
        atomic_list = list(set(atomic_list))
        qId2atomic[qId] = atomic_list
        atomic_num.append(len(atomic_list))
    atomic_num = np.array(atomic_num).mean()
    assert len(qId2atomic.keys()) == 480
    return qId2atomic, atomic_num 

def detect_initials(text):
    pattern = r"[A-Z]\. ?[A-Z]\."
    match = re.findall(pattern, text)
    return [m for m in match]

def run_gpt4(args, temp=0.7, top_p=0.9, frequency_penalty=0.0, presence_penalty=0.0, max_tokens=3):
    import openai
    OPENAI_API_KEY = args.openai_key
    openai.api_key = OPENAI_API_KEY
    prompt = f"* context: {_g_para}\n\n* statement:\n{atomic_facts}\n\nGenerate 'True' if all information in given statement is in given context. Else generate 'False'"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {'role': 'user', 'content': prompt}
        ],
        max_tokens=max_tokens,
        top_p=top_p,
        temperature=temp
    )
    answer = response['choices'][0]['message']['content']
    return answer

def cal_f1(p_score, g_score):
    if type(p_score) == bool:
        recall = 1 if p_score==True else 0
        prec = 1 if g_score==True else 0
    elif type(p_score) == int:
        recall = p_score 
        prec = g_score
    else:
        if len(p_score) == 0:
            recall = 0
        else:
            recall = p_score.count(1)/len(p_score)
        prec = g_score.count(1)/len(g_score)
    if prec + recall == 0:
        f1 = 0
    else:
        f1 = 2*prec*recall/(prec+recall)
    return prec, recall, f1

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--pred_file", type=str, required=True)
    parser.add_argument("--pred_atomic_file", type=str, required=True)
    parser.add_argument("--metric_model", type=str, default="cross", choices=['true', 'gpt4', 'bi', 'cross'])
    parser.add_argument("--threshold", type=float, default=6.0)
    parser.add_argument("--openai_key")
    parser.add_argument("--revised", action="store_true")
    args = parser.parse_args()

    pred_model = args.pred_model
    metric_model = args.metric_model

    if metric_model == "true":
        print(f"Loading TRUE tokenizer and model ...")
        tokenizer = AutoTokenizer.from_pretrained("google/t5_xxl_true_nli_mixture")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/t5_xxl_true_nli_mixture").cuda()
        print(f"Done Loading!\n\n")
    elif metric_model == "bi":
        model = SentenceTransformer('all-MiniLM-L6-v2')
    elif metric_model == "cross":
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
    elif metric_model == "gpt4":
        pass
    else:
        assert False

    print(f"Loading prediction result from .. {args.pred_file}")
    pred_f = json.load(open(args.pred_file))
    print(f'Loading atomic facts from .. {args.pred_atomic_file}')
    pred_atomic_facts, pred_atomic_num = get_pred_atomic_dict(json.load(open(args.pred_atomic_file)))

    """
    Get gold atomic facts
    """
    if args.revised:
        gold_atomic_path = "../data/revised_version.json"
    else:
        gold_atomic_path = "../data/original_version.json"
    gt_atomic_dict = json.load(open(gold_atomic_path))
    gt_atomic_facts = get_gold_atomic_dict(gt_atomic_dict) 

    qId_list = list(gt_atomic_dict.keys()) 
    assert len(qId_list) == 480, f"less than 480: {len(qId_list)}"

    f1_list = []; prec_list = []; recall_list = []
    
    _pred_name = args.pred_file.split("/")[-1].replace(".json", "")
    os.makedirs(f"scores/{args.metric_model}", exist_ok=True)
    save_path = f"scores/{args.metric_model}/{_pred_name}.json" 
    if os.path.exists(save_path) and args.metric_model not in ["bi", "cross"]:
        # Continue scoring
        print(f"Continue scoring from .. {save_path}")
        save_dict = json.load(open(save_path))
    else:
        print(f"Initial scoring!")
        save_dict = {} 

    for qId in tqdm(qId_list):
        if qId in save_dict: continue

        _p_atomic_facts = pred_atomic_facts[qId]
        _g_atomic_facts = gt_atomic_facts[qId]
        p_para = get_pred_output(pred_f[qId])

        _g_localDocList = gt_atomic_dict[qId]['context_list']
        _g_para = "\n".join(_g_localDocList)

        if metric_model == "true":
            qId_dict = {"atomic_fact": [], "paragraph": [], "p_score": [], "g_score": []}
            ## check if pred atomic fact is in gt paragraph
            for atomic_fact in _p_atomic_facts:
                p_score, p_logits = get_TRUE_score(_g_localDocList, atomic_fact) 
                qId_dict['atomic_fact'].append(atomic_fact)
                qId_dict['paragraph'].append(_g_para)
                qId_dict['p_score'].append(p_score)
            ## check if gt atomic fact is in pred paragraph
            for atomic_fact in _g_atomic_facts:
                g_score, g_logits = get_TRUE_score(p_para, atomic_fact)
                qId_dict['atomic_fact'].append(atomic_fact)
                qId_dict['paragraph'].append(p_para)
                qId_dict['g_score'].append(g_score)

        elif metric_model == "gpt4":
            qId_dict = {"atomic_facts": [], "paragraph": [], "output": [], "p_score": [], "g_score": []}
            p_score = []; g_score = []; results = []
            for atomic_facts in _p_atomic_facts:
                result = run_gpt4(args).lower()
                results.append(result)
                if "true" in result:
                    p_score.append(1)
                else:
                    p_score.append(0)
            qId_dict["atomic_facts"].append(_p_atomic_facts)
            qId_dict["paragraph"].append(_g_para)
            qId_dict["output"].append(results)
            qId_dict["p_score"].append(p_score)
            results = []
            for atomic_facts in _g_atomic_facts:
                result = run_gpt4(args).lower()
                results.append(result)
                if "true" in result:
                    g_score.append(1)
                else:
                    g_score.append(0) 
            qId_dict["atomic_facts"].append(_g_atomic_facts)
            qId_dict["paragraph"].append(p_para)
            qId_dict["output"].append(results)
            qId_dict["g_score"].append(g_score)

        elif metric_model == "bi":
            qId_dict = {"g_atomic_facts": [], "g_para": [], "p_atomic_facts": [], "p_para": [], "p_score": [], "g_score": []}
            assert len(_g_atomic_facts) > 0
            if len(_p_atomic_facts) == 0:
                p_score = [0]
                g_score = [0]
            else:
                g_score = cal_bi_score(_g_atomic_facts, p_para, args.threshold)
                p_score = cal_bi_score(_p_atomic_facts, _g_para, args.threshold)
            qId_dict["g_atomic_facts"].append(_g_atomic_facts)
            qId_dict["p_atomic_facts"].append(_p_atomic_facts)
            qId_dict["g_para"].append(_g_para)
            qId_dict["p_para"].append(p_para)
            qId_dict["p_score"].append(p_score)
            qId_dict["g_score"].append(g_score)

        elif metric_model == "cross":
            qId_dict = {"g_atomic_facts": [], "g_para": [], "p_atomic_facts": [], "p_para": [], "p_score": [], "g_score": []}
            assert len(_g_atomic_facts) > 0
            if len(_p_atomic_facts) == 0:
                p_score = [0]
                g_score = [0]
            else:
                g_score = list(cal_cross_score(p_para, _g_atomic_facts, args.threshold))
                p_score = list(cal_cross_score(_g_para, _p_atomic_facts, args.threshold))

            qId_dict["g_atomic_facts"].append(_g_atomic_facts)
            qId_dict["p_atomic_facts"].append(_p_atomic_facts)
            qId_dict["g_para"].append(_g_para)
            qId_dict["p_para"].append(p_para)
            qId_dict["p_score"].append(p_score)
            qId_dict["g_score"].append(g_score)

        else:
            assert False

        prec, recall, f1 = cal_f1(p_score, g_score)
        qId_dict['f1'] = f1
        qId_dict['recall'] = recall 
        qId_dict['prec'] = prec 
        save_dict[qId] = qId_dict

        if metric_model in ['gpt4', 'true']: 
            save_json(save_path, save_dict)

    f1_list = [elem['f1'] for elem in save_dict.values()]
    recall_list = [elem['recall'] for elem in save_dict.values()]
    prec_list = [elem['prec'] for elem in save_dict.values()]
    assert len(f1_list) == len(recall_list) == len(prec_list) == 480
    save_json(save_path, save_dict)

    print('='*80)
    if metric_model in ["bi", "cross"]:
        print(f"Threshold: {args.threshold}") 
    print(f'pred model: {pred_model}')
    print(f'Avg atomic num: {pred_atomic_num}')
    print(f'metric_model: {metric_model}')
    print(f'F1 score: {round(np.array(f1_list).mean()*100, 2)}')
    print(f'Prec score: {round(np.array(prec_list).mean()*100, 2)}')
    print(f'Recall score: {round(np.array(recall_list).mean()*100, 2)}')
    print(f"Saving in .. {save_path}")
    print('='*80)


