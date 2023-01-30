import openai

API_KEY = '' # put here your API KEY
openai.api_key = API_KEY

model = 'text-davinci-003'


sentence = str(input('Chat: '))

response = openai.Completion.create(
    prompt=sentence,
    model=model,
    max_tokens=800, #more tokens, more words, more $$
    temperature=0.4, # ai "creativity" level: 0 to 1
    n=1 # number of results
)

for result in response.choices:
    print(result.text)
