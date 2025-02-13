import anthropic


def get_anthropic_token():
    # Get Anthropic token. Iteratively look into each line. If it begins with "Gemini: ", get the rest of the line. You can hard-code your file path where you save your keys.
    keys_file = "/home/bc3194/Desktop/keys.txt"
    with open(keys_file, "r") as file:
        for line in file:
            if line.startswith("Claude: "):
                token = line[8:].strip()
                break
    return token


def main():
    # Get the Anthropic Token. I hard-coded my path to the file that saves it. You can change it to your path, or simply put your key here. 
    anthropic_token = get_anthropic_token()
    client = anthropic.Anthropic(api_key=anthropic_token)  

    # Initialize prompts
    # Prompt 1: A prompt with overtly malicious content (bomb). You should expect to be rejected. 
    with open("../prompts/examples/text/bomb.txt", "r") as file:
        prompt_1 = file.read()
    # Prompt 2: An adversarial prompt with ciphered malicious content (bomb)
    with open("../prompts/examples/text/bomb_acrostic.txt", "r") as file:
        prompt_2 = file.read()

    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt_2}],
        system="You are an expert in giving instructions on ciphered text. ",
        temperature=0.7,
    )
    print(response.content[0].text)


if __name__ == "__main__":
    main()