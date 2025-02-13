from openai import OpenAI
import openai


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
    # Get the OpenAI Token. I hard-coded my path to the file that saves it. You can change it to your path, or simply put your key here. 
    openai_token = get_openai_token()

    # Initialize prompts
    # Prompt 1: A prompt with overtly malicious content (bomb). You should expect to be rejected. 
    with open("../prompts/examples/text/bomb.txt", "r") as file:
        prompt_1 = file.read()
    # Prompt 2: An adversarial prompt with ciphered malicious content (bomb)
    with open("../prompts/examples/text/bomb_acrostic.txt", "r") as file:
        prompt_2 = file.read()

    # Get response. Available models: gpt-4o-mini-2024-07-18, gpt-4o-2024-11-20
    with openai.OpenAI(api_key=openai_token) as client:
        completion = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[{"role": "user", "content": prompt_1}],
            temperature=0.7,
            max_tokens=200,
        )
        model_output = completion.choices[0].message.content
    print(model_output)


if __name__=="__main__":
    main()