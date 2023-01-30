import openai

API_KEY = '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'
openai.api_key = API_KEY

model = 'text-davinci-003'


sentence = str(input('Chat: '))

response = openai.Completion.create(
    prompt=sentence,
    model=model,
    max_tokens=800,
    temperature=0.4,
    n=2
)

for result in response.choices:
    print(result.text)
