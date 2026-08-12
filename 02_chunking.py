from langchain_community.document_loaders import PyPDFLoader

# غيّر الاسم ده باسم الملف اللي رفعته فعلاً
loader = PyPDFLoader("اسم_الملف.pdf")
pages = loader.load()

print("عدد الصفحات اللي اتحمّلت:", len(pages))
print("--- عينة من أول صفحة ---")
print(pages[0].page_content[:500])

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # كل chunk حوالي 500 حرف
    chunk_overlap=50,    # تداخل 50 حرف بين كل chunk والتاني عشان المعنى ميضيعش
    separators=["\n\n", "\n", ". ", " ", ""]  # بيحاول يقطع عند حدود منطقية الأول
)

chunks = text_splitter.split_documents(pages)

print("عدد الـ chunks اللي اتعملت:", len(chunks))
print("--- عينة من أول chunk ---")
print(chunks[0].page_content)
print("--- عينة من تاني chunk ---")
print(chunks[1].page_content)