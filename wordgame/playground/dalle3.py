from openai import OpenAI
import openai


def get_openai_token():
    # Get Openai token. Iteratively look into each line. If it begins with "OpenAI: ", get the rest of the line. You can hard-code your file path where you save your keys.
    keys_file = "/home/bc3194/Desktop/keys.txt"
    with open(keys_file, "r") as file:
        for line in file:
            if line.startswith("OpenAI: "):
                openai_token = line[8:-1]
                break
    return openai_token


def main():
    # Get the OpenAI Token. I hard-coded my path to the file that saves it. You can change it to your path, or simply put your key here. 
    openai_token = get_openai_token()

    # Initialize prompts
    # Prompt 0: A fully normal prompt
    prompt_0 = "Draw me an image of a lovely cat."
    # Prompt 1: A prompt with a deduction test
    prompt_1 = "What is the letter after g? Draw that in capital please. " 
    # Prompt 2: A prompt with overtly malicious content (gun). You should expect to be rejected. 
    with open("../prompts/examples/gun.txt", "r") as file:
        prompt_2 = file.read()
    # Prompt 3: An adversarial prompt with ciphered malicious content (gun)
    with open("../prompts/examples/gun_acrostic.txt", "r") as file:
        prompt_3 = file.read()

    # Get response
    client = OpenAI(api_key=openai_token)
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt_2,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    print(response.data[0].url)


if __name__=="__main__":
    main()