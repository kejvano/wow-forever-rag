import os

# ask.py and index.py build an OpenAI client at import time; no test ever calls it
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")