import os
import json
import random
import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig
from tqdm import tqdm
from argparse import ArgumentParser

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def write_json_file(file_path, out):
    with open(file_path, 'w') as f:
        json.dump(out, f, indent='\t')  

def generate_prompt(context, qText, is_tulu):
    instruction = "Generate an [answer] to the given [question] in full sentence by utilizing all necessary information in given [context] and limiting the utilized information to that [context]. Provide all information you utilize from given [context] to answer the question."

    instruction = f"{instruction}\n[contexts]\n{context}\n[question]\n{qText}\nDon't Forget that you have to generate an [answer] to the given [question] in full sentence by utilizing all necessary information in given [context] and information only from the [context]. Also, provide all information you utilize from given [context]\n[answer]" 

    if is_tulu:
        return f"<|user|>\n{instruction}\n<|assistant|>\n"
    else:
        return f"{instruction}\n"

def main(args):

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        load_in_4bit=True,
        device_map='auto',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()    
    tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, 
            use_fast=False,
    )
    
    if args.model_name in ["vicuna", "tulu"]:
        tokenizer.padding_side = 'left'   

    if args.model_name == "tulu":
        generation_config = GenerationConfig(
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=0.6,
            do_sample=True,
            early_stopping=True,
            num_return_sequences=1
        )
    elif args.model_name in ["falcon", "llama2", "vicuna"]:
        generation_config = GenerationConfig(
            max_length= 4096, 
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id
        )
    else:
        assert False

    data = read_json_file(args.data)
   
    _model_path = args.model_path.split('/')[-1]    
    _data_type = args.data.split("/")[-1].split("_")[0]
    assert _data_type in ['original', 'revised']

    save_path = f"../results/{_data_type}.add_distractor_{args.add_distractor}.no_instruction_{args.no_instruction}.{_model_path}.{args.distractor_max_seq_length}.json"
    if os.path.exists(save_path):
        qId2answer = read_json_file(save_path)
        data_iter = len(qId2answer)
        print(f"Loading file from .. {save_path}\nStarting from {data_iter}")
    else:
        data_iter = 0
        qId2answer = {}

    for qId in tqdm(data.keys()):
        if qId in qId2answer: 
            continue
        
        v_dict = data[qId]
        assert qId == v_dict["qId"]
        qText = v_dict['qText']
        context_list = v_dict["context_list"] 

        if args.add_distractor:
            qId2distractor = read_json_file(args.distractor_data)
            distractor_list = qId2distractor[qId][f"{args.distractor_max_seq_length}_distractor"]

            if args.distractor_place == "start":
                context_list = distractor_list + context_list 
            elif args.distractor_place == "end":
                context_list = context_list + distractor_list 
            elif args.distractor_place == "random":
                context_list = context_list + distractor_list 
                random.shuffle(context_list)
            else: 
                assert False

        context = "\n".join(context_list)

        if args.no_instruction:
            input_entry = qText 
        else:
            input_entry = generate_prompt(context, qText, is_tulu=(args.model_name=="tulu"))
        
        input_ids = tokenizer(
            input_entry, 
            max_length=args.distractor_max_seq_length if args.add_distractor else args.max_seq_length, 
            return_tensors="pt").input_ids
        input_ids = input_ids.to('cuda')
        outputs = model.generate(
            input_ids = input_ids,
            generation_config = generation_config,
            max_new_tokens = args.distractor_max_seq_length if args.add_distractor else args.max_seq_length, 
            return_dict_in_generate = True
        )
        outputs_text = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        qId2answer[qId] = {'input': input_entry, 'ids_shape': input_ids.shape[-1], 'output': outputs_text}

        if data_iter % 50 == 0:
            write_json_file(save_path, qId2answer)

    print(f'Saving in .. {save_path}')
    write_json_file(save_path, qId2answer)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="vicuna", choices=['vicuna', 'tulu', 'llama2', 'falcon'])
    parser.add_argument("--model_path", type=str, default="lmsys/vicuna-33b-v1.3")
    parser.add_argument("--data", type=str, default="../data/original_version.json")
    parser.add_argument("--no_instruction", action="store_true")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    ### original-dist, revised-dist
    parser.add_argument("--add_distractor", action="store_true")
    parser.add_argument("--distractor_data", type=str, default="../data/qId2distractor.json")
    parser.add_argument("--distractor_max_seq_length", type=str, default="2k")
    parser.add_argument("--distractor_place", type=str, default="random", choices=['start', 'end', 'random'])
    args = parser.parse_args()

    if args.add_distractor:
        assert args.distrator_max_seq_length in ["1k", "2k", "3k", "4k", "8k", "12k", "16k"]

    print(args)
    main(args)
