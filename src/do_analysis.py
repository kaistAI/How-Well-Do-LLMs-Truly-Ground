import json
import numpy as np

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
    return prec*100, recall*100, f1*100

qId2pop = json.load(open("id2pop.json"))
qId2type = json.load(open("id2split.json")) 

#fname = "results2/no_instruction_False.vicuna-7b-v1.5/cross_nli_para.json"
#fname = "results2_revised/no_instruction_False.tulu-65b-recovered/cross_nli_para.json"
fname = "results2_distractor/no_instruction_False.Llama-2-13b-hf.2048/cross_nli_para.json"
#fname = "results2_revised_distractor/no_instruction_False.tulu-65b-recovered.2048/cross_nli_para.json"
print(f'### Loading file .. {fname}')
f = json.load(open(fname))

qId_list = list(f.keys())

f1_pop_dict = {'high': [], 'low': []}
f1_type_dict = {'min': [], 'max': [], 'multi': []}
prec_pop_dict = {'high': [], 'low': []}
prec_type_dict = {'min': [], 'max': [], 'multi': []}
recall_pop_dict = {'high': [], 'low': []}
recall_type_dict = {'min': [], 'max': [], 'multi': []}

for qId in qId_list:
    p_score = f[qId]['p_score'][0]
    g_score = f[qId]['g_score'][0]
    prec, recall, f1 = cal_f1(p_score, g_score)
    # print(f"prec: {prec} || recall: {recall} || f1: {f1}")
    # import sys; sys.exit()
    f1_pop_dict[qId2pop[qId]].append(f1)
    f1_type_dict[qId2type[qId]].append(f1)
    prec_pop_dict[qId2pop[qId]].append(prec)
    prec_type_dict[qId2type[qId]].append(prec)
    recall_pop_dict[qId2pop[qId]].append(recall)
    recall_type_dict[qId2type[qId]].append(recall)

print("### F1 ###")
total_f1 = list(f1_pop_dict["low"]) + list(f1_pop_dict["high"])
total_f1 = round(np.array(total_f1).mean(), 2)
print(f"Total F1 {total_f1}")
for pop, f1_list in f1_pop_dict.items():
    f1_score = round(np.array(f1_list).mean(), 2)
    print(f"[{pop}] {f1_score}")

for _type, f1_list in f1_type_dict.items():
    f1_score = round(np.array(f1_list).mean(), 2)
    print(f"[{_type}] {f1_score}")

print("### Prec ###")
total_prec = list(prec_pop_dict["low"]) + list(prec_pop_dict["high"])
total_prec = round(np.array(total_prec).mean(), 2)
print(f"Total Prec {total_prec}")

for pop, prec_list in prec_pop_dict.items():
    prec_score = round(np.array(prec_list).mean(), 2)
    print(f"[{pop}] {prec_score}")

for _type, prec_list in prec_type_dict.items():
    prec_score = round(np.array(prec_list).mean(), 2)
    print(f"[{_type}] {prec_score}")

print("### Recall ###")
total_recall = list(recall_pop_dict["low"]) + list(recall_pop_dict["high"])
total_recall = round(np.array(total_recall).mean(), 2)
print(f"Total Recall {total_recall}")

for pop, recall_list in recall_pop_dict.items():
    recall_score = round(np.array(recall_list).mean(), 2)
    print(f"[{pop}] {recall_score}")

for _type, recall_list in recall_type_dict.items():
    recall_score = round(np.array(recall_list).mean(), 2)
    print(f"[{_type}] {recall_score}")


