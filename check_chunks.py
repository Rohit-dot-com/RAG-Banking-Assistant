import warnings
warnings.filterwarnings("ignore")
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pages = PyPDFLoader("data/sample.pdf").load()
for size in [800, 1000, 1200, 1500]:
    for overlap in [100, 150, 200]:
        n = len(RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap).split_documents(pages))
        print(size, overlap, "->", n, "chunks")