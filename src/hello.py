import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads .env into environment variables

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

response = client.chat.completions.create(
    model="gpt-4o-mini",  # cheap and fast, good enough for everything in this project
    messages=[
        {"role": "system", "content": "Only answer if you are certain. Otherwise say you don't know."},
        {"role": "user", "content": "What is World of Warcraft Forever?"},
    ],
)

print(response.choices[0].message.content)