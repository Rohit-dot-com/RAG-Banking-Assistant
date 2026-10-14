import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()  # reads .env and makes GOOGLE_API_KEY available

# Model names change over time. If you get a 404 / NOT_FOUND error, open
# Google AI Studio's model list and copy a current "flash" model name here.
MODEL_NAME = "gemini-2.5-flash"

def as_text(content):
    # Some models return a list of parts instead of a plain string
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)

llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)
response = llm.invoke("In one sentence, what is a safe-haven asset?")
print(as_text(response.content))