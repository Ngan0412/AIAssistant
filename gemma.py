from openai import OpenAI

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)
# response = client.chat.completions.create(
#   model="gemma3:4b",
#   messages=[
#     {"role": "user", "content": "hello"}
#   ]
# )
# reply = response.choices[0].message.content
# print(reply)
messages =[]
while True:
    user_input = input("\nYou: ")
    if user_input =="exit":
        break
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
          model="gemma3:4b",
          stream=True,
          messages=messages
    )
    bot_reply = ""
    for chunk in response:
        bot_reply += chunk.choices[0].delta.content or ""
        print(chunk.choices[0].delta.content or "", end="",flush=True)
    messages.append({"role": "assistant", "content": bot_reply})