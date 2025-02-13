import google.generativeai as genai
from google.generativeai import types


def get_google_token():
    # Get Google token. Iteratively look into each line. If it begins with "Gemini: ", get the rest of the line. You can hard-code your file path where you save your keys.
    keys_file = "/home/bc3194/Desktop/keys.txt"
    with open(keys_file, "r") as file:
        for line in file:
            if line.startswith("Gemini: "):
                token = line[8:].strip()
                break
    return token


def main():
    # Get the Google Token. I hard-coded my path to the file that saves it. You can change it to your path, or simply put your key here. 
    google_token = get_google_token()

    # Initialize prompts
    # Prompt 1: A prompt with overtly malicious content (bomb). You should expect to be rejected. 
    with open("../prompts/examples/text/bomb.txt", "r") as file:
        prompt_1 = file.read()
    # Prompt 2: An adversarial prompt with ciphered malicious content (bomb)
    with open("../prompts/examples/text/bomb_acrostic.txt", "r") as file:
        prompt_2 = file.read()

    genai.configure(api_key=google_token)
    model = genai.GenerativeModel("gemini-1.5-pro")
    generation_config={
        "max_output_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "stop_sequences": ["\n\n\n"]
    }
    response = model.generate_content(prompt_2, generation_config=generation_config)
    print(response.text)


if __name__ == "__main__":
    main()