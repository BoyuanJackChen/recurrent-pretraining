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

    ### Load hint generation templates and wordgame template
    with open("../prompts/word_game_generation.txt", "r") as file:
        hint_template = file.read()
    with open("../prompts/wordgame.txt", "r") as file:
        wordgame_template = file.read()

    ### Load keywords dataset with pandas
    keywords_file = "../outputs/keywords.csv"
    keywords = pd.read_csv(keywords_file)

    ### Initialize output dataframe with pandas
    output_df = pd.DataFrame(columns=["malicious_token", "query", "wordgame"])
    if not os.path.exists(keywords_file):
        output_df.to_csv(keywords_file, index=False)
    else:
        num_already_generated = pd.read_csv(keywords_file).shape[0]

    ### Iterate over each row in keywords and generate word hints and wordgame prompts
    for i, row in keywords.iterrows():
        ### Check if the current prompt already exists
        if i < num_already_generated:
            continue

        ### Initialize the variables
        query = row["query"]
        malicious_word = row["malicious_token"]
        masked_query = query.replace(malicious_word, "[MASK]")

        ### Generate the hints for the malicious_word
        while True:
            with openai.OpenAI(api_key=openai_token) as client:
                completion = client.chat.completions.create(
                    model="gpt-4o-2024-11-20",
                    messages=[{"role": "user", "content": hint_template.replace("{word}", malicious_word)}],
                    temperature=0.7,
                    max_tokens=512,
                )
                hints = completion.choices[0].message.content.strip().lower()

        ### Put query and hints into the wordgame template
        final_prompt = wordgame_template.replace("{masked_query}", masked_query).replace("{word_game}", hints)
        print(query)
        print(final_prompt)
        input()
        final_prompt = final_prompt.replace("\n", "\\n")
        
        ### Read and write the update to output with concatenate
        output_df = pd.read_csv("../outputs/wordgame.csv")
        output_df = pd.concat([output_df, pd.DataFrame({"malicious_token": [malicious_word], "query": [final_query], "wordgame": [final_prompt]})], ignore_index=True)
        output_df.to_csv("../outputs/wordgame.csv", index=False)


if __name__=="__main__":
    main()