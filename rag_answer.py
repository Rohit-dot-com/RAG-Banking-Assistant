import time
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

DB_DIR = "chroma_db"
MODEL_NAME = "gemini-2.5-flash-lite"   # use the same name that worked in test_llm.py
TOP_K = 5

# 1) Load the saved vector store and turn it into a retriever
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
if not vectorstore.get(limit=1)["ids"]:
    raise SystemExit("Vector store is empty. Run embed_and_search.py first.")
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

# 2) The LLM
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)

# 3) The grounded prompt
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a careful assistant that answers questions about a document.\n"
     "Rules:\n"
     "1. Use ONLY the context below. Do not use outside knowledge.\n"
     "2. If the context does not contain the answer, reply exactly: "
     "\"I could not find this in the document.\"\n"
     "3. Be concise and cite the page numbers you used, like (p. 12).\n"
     "4. If the context describes other researchers' work (for example a literature "
     "review), say so. Never present other studies' methods or findings as this "
     "document's own.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])

def as_text(content):
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)

def ask(question):
    # Retrieve the top-k chunks
    docs = retriever.invoke(question)
    # Build the context: each chunk labeled with its page
    context = "\n\n".join(f"[Page {d.metadata['page']}] {d.page_content}" for d in docs)
    # Fill the prompt and call the LLM
    messages = prompt.invoke({"context": context, "question": question})
    response = llm.invoke(messages)
    # Unique source pages, in retrieval order
    pages = list(dict.fromkeys(d.metadata["page"] for d in docs))
    return as_text(response.content), pages

if __name__ == "__main__":
    questions = [
        "What happens to gold prices when geopolitical risk rises?",
        "What is the sample period and frequency of the data used in the paper?",
        "What is the capital of France?",
        "Who is the current RBI governor?",
        "What data and methods do the authors use?",
    ]
    for q in questions:
        answer, pages = ask(q)
        print("\n" + "=" * 70)
        print("QUESTION:", q)
        print("ANSWER:", answer)
        print("RETRIEVED PAGES:", ", ".join(pages))
        time.sleep(2)