""" Boyuan
Attemptive implementation of the recurrent transformer using their instructions on huggingface
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def main():
    ### Initialize model, tokenizer and config
    device = torch.device("cuda:0")
    model = AutoModelForCausalLM.from_pretrained(
        "tomg-group-umd/huginn-0125", 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True,
        device_map=device
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")

    config = GenerationConfig(
        max_new_tokens=256, 
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

    ### Safety template loaded from ./raw_safety_template.txt
    with open("raw_half_safety.txt", "r") as f:
        safety_template = f.read()
    
    ### Malicious prompt loaded from ../prompts/bomb_acrostic.txt
    with open("../prompts/bomb_wordgame.txt", "r") as f:
        prompt = f.read()

    ### Generate the output in server-like format
    messages = []
    messages.append({"role": "system", "content": safety_template})
    messages.append({"role": "user", "content": prompt})
    chat_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(chat_input, return_tensors="pt", add_special_tokens=False).to(model.device)
    output_ids = model.generate(input_ids, config, tokenizer=tokenizer, num_steps=4)
    output_text = tokenizer.decode(output_ids.sequences[0][input_ids.shape[1]:], skip_special_tokens=True)

    ### Print output text
    print(output_text)



if __name__ == "__main__":
    main()