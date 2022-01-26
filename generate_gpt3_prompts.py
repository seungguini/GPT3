import os
import openai
from dotenv import load_dotenv
import pickle
import requests

# This is just Prof. Sedoc's example for code generation
# params = {
#     'NetId': 'js11531',
#     'prompt' : """
# Dataframe df1 has three columns date, name, zipcode
# Dataframe df2 has four columns zipcode, restaurants, average price, average stars
# We join df1 znd df2 by zipcode
# Finally we order the table by average stars
#     """,
#     'maxtokens' : 1024
# }
# URL = 'http://128.122.85.143/codex'
# resp = requests.post(URL, data="",params=params)


load_dotenv()
openai.api_key = os.getenv("API_KEY")

DATASET_NAME = "test"
FOLDER_PATH = "./model_responses"

# Convert <s> style prompts to OpenAI prompts

prompt_list = []

with open(f'{DATASET_NAME}_prompts.txt', 'r') as f:
    for line in f:
        prompt_list.append(line)

gpt3_prompts = []

for prompt in prompt_list:

    gpt3_prompt = "The following is a conversation between Person1 and Person2.\n"

    utterances = prompt.split("</s>")
    person1 = True
    for utterance in utterances:
        person = "\nPerson1: " if person1 else "\nPerson2: "
        gpt3_prompt = gpt3_prompt + person + utterance
        person1 = not person1

    # Add final Person (no trailing space)
    person = "Person1:" if person1 else "Person2:"
    gpt3_prompt += person

    gpt3_prompts.append(gpt3_prompt)

# Generate responses
start_sequence = "\nPerson1:"
restart_sequence = "\nPerson2: "

responses = []

for prompt in gpt3_prompts:
    print(prompt)
    print('-------')
    response = openai.Completion.create(
    engine="text-davinci-001",
    #prompt="The following is a conversation between two people.\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?\nHuman: I'd like to cancel my subscription.\nAI:",
    prompt=prompt,
    temperature=0.9,
    max_tokens=150,
    top_p=0.92,
    frequency_penalty=0.0,
    presence_penalty=0.6,
    )
    text = response.get("choices")[0].get("text") 
    
    if text == "":
        text = "no reponse"
    responses.append(text)
    print('response:', text)

# Save responses in txt file
with open(f"./model_responses/{DATASET_NAME}_gpt3_response.txt", "w") as f:
    for response in responses:
        f.write(response + "\n")

# Save response list
with open(f'./model_responses/{DATASET_NAME}_gpt3_responses.pkl', 'wb') as f:
    pickle.dump(responses, f)

