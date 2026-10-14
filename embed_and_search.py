import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PDF_PATH = "data/sample.pdf"
DB_DIR = "chroma_db"

# 1) Load and split (same as step 1)
pages = PyPDFLoader(PDF_PATH).load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
chunks = splitter.split_documents(pages)

# 2) Clean the metadata: keep only the file name and the printed page number
for c in chunks:
    c.metadata = {
        "source": os.path.basename(c.metadata.get("source", PDF_PATH)),
        "page": c.metadata.get("page_label", str(c.metadata.get("page", 0) + 1)),
    }

# 3) Load the embedding model (downloads on first run)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Peek at what an embedding looks like
vec = embeddings.embed_query("gold is a safe haven asset")
print(f"Vector length: {len(vec)}")
print(f"First 5 numbers: {[round(x, 3) for x in vec[:5]]}")

# 4) Build the vector store once, then reuse it (rebuild if it is empty)
vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
if vectorstore.get(limit=1)["ids"]:
    print("\nLoaded existing vector store")
else:
    vectorstore.add_documents(chunks)
    print(f"\nCreated vector store with {len(chunks)} chunks")

# 5) Similarity search
def search(question, k=3):
    print("\n" + "=" * 70)
    print("QUESTION:", question)
    results = vectorstore.similarity_search_with_score(question, k=k)
    for rank, (doc, distance) in enumerate(results, start=1):
        print(f"\n[{rank}] distance={distance:.3f} | page {doc.metadata['page']}")
        print(doc.page_content[:300].replace("\n", " "))

search("What happens to gold prices when geopolitical risk rises?")
search("What data and methods do the authors use?")
search("What is the capital of France?")
search("What is the sample period and frequency of the data used in the paper?", k=5)
search("How do the authors measure geopolitical risk?", k=5)