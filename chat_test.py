import os

import openai


def load_env():
    with open('secrets.env') as file:
        for line in file:
            key, value = line.strip().split('=')
            os.environ[key] = value


load_env()
openai.api_key = os.environ['OPENAI_API_KEY']

AI_ENGINE = 'text-davinci-003'


def main(preamble):
    conversation = preamble

    while True:
        new_text = input("S: ")
        if new_text == 'q':
            break

        conversation += new_text + "RD: "

        # Perform a completion from openai
        completion = openai.Completion.create(
            model=AI_ENGINE,
            prompt=conversation,
            temperature=0.9,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0.6,
            stop=["S: ", "RD: "],
        )
        response = completion.choices[0].text.strip()
        print("RD:", response)
        conversation += response + "S: "


if __name__ == '__main__':
    # main("When RD shares code, he wraps it in a code fence (e.g. '```python <code> ```')\nS: ")
    with open("prompts/duck-pond.txt") as file:
        main(file.read())
