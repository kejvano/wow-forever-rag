import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Only answer if you are certain. Otherwise say you don't know."},
        {"role": "user", "content": "What is World of Warcraft Forever?"},
    ],
)

print(response.choices[0].message.content)