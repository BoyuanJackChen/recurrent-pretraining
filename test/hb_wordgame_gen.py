""" Boyuan
Attemptive implementation of the recurrent transformer using their instructions on huggingface
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import pandas as pd
import os
import gc
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--r", type=int, default=64)
FLAGS = parser.parse_args()


def main(args):
    ### Initialize model, tokenizer, config and recurrent hyperparameter
    device = torch.device("cuda:0")
    model_name = "tomg-group-umd/huginn-0125"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True,
        device_map=device
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = GenerationConfig(
        max_new_tokens=512, 
        stop_strings=["<|end_text|>", "<|end_turn|>"], 
        use_cache=True,
        do_sample=False, 
        temperature=None, 
        top_k=None, 
        top_p=None, 
        min_p=None, 
        return_dict_in_generate=True,
        eos_token_id=65505,
        bos_token_id=65504,
        pad_token_id=65509
    )
    r = args.r

    ### Safety template loaded from ./raw_safety_template.txt
    with open("raw_safety_template.txt", "r") as f:
        safety_template = f.read()
        
    ### Load the wordgame-adapted harmbench dataset
    wordgame = pd.read_csv("../prompts/hb_wordgame.csv")
    
    ### Create a pandas df with four columns: r, answer, question, malicious_token
    outfile = f"../outputs/raw_r{r}.csv"
    df = pd.DataFrame(columns=["r", "answer", "question", "malicious_token"])
    if not os.path.exists(outfile):
        df.to_csv(outfile, index=False)

    ### Generate the output in server-like format for each question in wordgame. Use zip
    for (harmprompt, malicious_token) in wordgame[["prompt", "malicious_token"]].values:
        ### Check if the current prompt already exists
        df = pd.read_csv(outfile)
        if harmprompt in df["question"].values:
            print(f"{len(df)}: {harmprompt} already exists.")
            continue

        ### Generate output
        messages = []
        messages.append({"role": "system", "content": safety_template})
        messages.append({"role": "user", "content": harmprompt})
        chat_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(chat_input, return_tensors="pt", add_special_tokens=False).to(model.device)
        output_ids = model.generate(input_ids, config, tokenizer=tokenizer, num_steps=r)
        output_text = tokenizer.decode(output_ids.sequences[0][input_ids.shape[1]:], skip_special_tokens=True)
        output_text = output_text.replace("\n", "\\n")
        
        ### Write output to the output df with concatenate function
        df = pd.concat([df, pd.DataFrame({"r": [r], "answer": [output_text], "question": [harmprompt], "malicious_token": malicious_token})])
        df.to_csv(outfile, index=False)
        
        ### Print output text
        print(f"{len(df)}: {output_text}")
        
        ### Clear model's cache and memory
        del output_ids
        del input_ids
        torch.cuda.empty_cache()
        gc.collect()



if __name__ == "__main__":
    main(FLAGS)