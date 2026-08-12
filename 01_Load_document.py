# !pip install -q langchain langchain-community chromadb sentence-transformers transformers accelerate pypdf
from google.colab import files
uploaded = files.upload()

from langchain_community.document_loaders import PyPDFLoader

# Change the name of the PDF file to match the uploaded file name
loader = PyPDFLoader(list(uploaded.keys())[0])
pages = loader.load()

print("The Number of Pages Loaded:", len(pages))
print("Sample Content :")
print(pages[0].page_content[:500])