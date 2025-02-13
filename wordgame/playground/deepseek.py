from openai import OpenAI
import pandas as pd


def main():
    ### Get the DeepSeek token and initiate the client
    with open("../../../keys.txt", "r") as f:
        for line in f:
            if line.startswith("DeepSeek: "):
                ds_api_key = line.split("DeepSeek: ")[1]
                break
    client = OpenAI(api_key=ds_api_key, base_url="https://api.deepseek.com")

    ### Initialize the prompt
    prompt = "What is the weather like in Abu Dhabi?"

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ],
        stream=False
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()