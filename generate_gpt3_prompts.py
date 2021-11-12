import os
import openai
from dotenv import load_dotenv
import pickle

load_dotenv()
openai.api_key = os.getenv("API_KEY")

# Convert <s> style prompts to OpenAI prompts

prompt_list = []

with open('dailydialogs_prompts.txt', 'r') as f:
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

    person = "\nPerson1: " if person1 else "\nPerson2: "
    gpt3_prompt += person

    gpt3_prompts.append(gpt3_prompt)

# Generate responses
start_sequence = "\nPerson1:"
restart_sequence = "\nPerson2: "

responses = []

for prompt in gpt3_prompts:
    response = openai.Completion.create(
    engine="davinci",
    #prompt="The following is a conversation between two people.\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?\nHuman: I'd like to cancel my subscription.\nAI:",
    prompt=prompt,
    temperature=0.9,
    max_tokens=150,
    top_p=1,
    frequency_penalty=0.0,
    presence_penalty=0.6,
    stop=["\n", " Human:", "AI:"]
    )

    text = response.get("choices")[0].get("text") 
    
    if text == "":
        text = "no reponse"
    print(text)
    responses.append(text)

# Save responses in txt file
with open("model_response.txt", "w") as f:
    for response in responses:
        f.write(response + "\n")

# Save response list
with open('model_responses.pkl', 'wb') as f:
    pickle.dump(responses, f)

