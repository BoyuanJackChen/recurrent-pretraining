import openai
import pandas as pd
import os


def get_openai_token():
    # Get Openai token. Iteratively look into each line. If it begins with "OpenAI: ", get the rest of the line. You can hard-code your file path where you save your keys.
    keys_file = "/home/bc3194/Desktop/keys.txt"
    with open(keys_file, "r") as file:
        for line in file:
            if line.startswith("OpenAI: "):
                token = line[8:].strip()
                break
    return token


def main():
    ### Get the OpenAI Token. 
    openai_token = get_openai_token()

    ### Load identification and rewrite prompt from WordGame
    with open("../prompts/malicious_word_identification.txt", "r") as file:
        template = file.read()
    with open("../prompts/rewrite.txt", "r") as file:
        rewrite_template = file.read()

    ### Load Harmbench dataset with pandas
    harmbench = pd.read_csv("../../prompts/harmful_behaviors.csv")

    ### Initialize output dataframe with pandas
    output_df = pd.DataFrame(columns=["malicious_token", "query", "rewrite"])
    if not os.path.exists("../outputs/keywords.csv"):
        output_df.to_csv("../outputs/keywords.csv", index=False)
    else:
        num_already_generated = pd.read_csv("../outputs/keywords.csv").shape[0]

    ### Iterate over each row in harmbench and generate malicious tokens and rewrites
    for i, row in harmbench.iterrows():
        if i < num_already_generated:
            continue
        query = row["goal"]
        prompt = template.replace("{query}", query)

        ### Detect most malicious with openai models. Available checkpoints: gpt-4o-mini-2024-07-18, gpt-4o-2024-11-20
        while True:
            with openai.OpenAI(api_key=openai_token) as client:
                completion = client.chat.completions.create(
                    model="gpt-4o-2024-11-20",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=10,
                )
                malicious_word = completion.choices[0].message.content.strip().lower()
            # Make sure that malicious_token is one word
            if (not (" " in malicious_word)) and malicious_word in query.lower():
                print(f"Malicious token for query '{query}' is '{malicious_word}'")
                break

        ### Make sure that malicious_word is a unique word in query. Exclude the cases where malicious_word is a substring of another word. (No rewrite was required in harmbench prompts)
        rewrite = False
        if malicious_word in query.lower() and query.lower().split(" ").count(malicious_word) <= 1:
            harmbench.at[i, "malicious_token"] = malicious_word
            final_query = query
        else: 
            rewrite = True
            prompt = rewrite_template.replace("{query}", query).replace("{malicious_word}", malicious_word)
            while True:
                with openai.OpenAI(api_key=openai_token) as client:
                    completion = client.chat.completions.create(
                        model="gpt-4o-2024-11-20",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        max_tokens=10,
                    )
                    final_query = completion.choices[0].message.content.strip()
                    print(f"Rewrite is '{final_query}'")
                if final_query.lower().split().count(malicious_word) == 1:
                    print(f"Rewrite is '{final_query}'")
                    break
        
        ### Read and write the update to output with concatenate
        output_df = pd.read_csv("../outputs/keywords.csv")
        output_df = pd.concat([output_df, pd.DataFrame({"malicious_token": [malicious_word], "query": [final_query], "rewrite": [rewrite]})], ignore_index=True)
        output_df.to_csv("../outputs/keywords.csv", index=False)


if __name__=="__main__":
    main()