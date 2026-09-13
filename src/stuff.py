import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

with open("data/announcement.txt", encoding="utf-8") as f:
    document = f.read()

SYSTEM = """You answer questions about the game World of Warcraft Forever.
Use ONLY the information in the provided source. If the source does not
contain the answer, reply exactly: "I don't have information about that."
Never use prior knowledge."""

question = "When does it release?"

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"SOURCE:\n{document}\n\nQUESTION: {question}"},
    ],
)

print(response.choices[0].message.content)
print(response.usage)