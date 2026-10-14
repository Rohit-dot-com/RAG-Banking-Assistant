from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "data/sample.pdf"

# 1) LOAD: read the PDF into a list of Document objects (one per page)
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

print(f"Pages loaded: {len(pages)}")
print("\n--- First page preview (300 characters) ---")
print(pages[0].page_content[:300])
print("\n--- Metadata of first page ---")
print(pages[0].metadata)

# 2) SPLIT: cut the pages into overlapping chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # max characters per chunk
    chunk_overlap=150,   # characters shared between neighboring chunks
)
chunks = splitter.split_documents(pages)

lengths = [len(c.page_content) for c in chunks]
print(f"\nChunks created: {len(chunks)}")
print(f"Average chunk length: {sum(lengths) / len(lengths):.0f} characters")
print(f"Shortest / longest: {min(lengths)} / {max(lengths)}")

# 3) INSPECT: look at one chunk and its metadata
i = min(4, len(chunks) - 1)
print(f"\n--- Chunk {i + 1} ---")
print(chunks[i].page_content)
print("\nMetadata:", chunks[i].metadata)