from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"  # هيتخزن على الـ disk بتاع Colab
)

print("تم تخزين", vectorstore._collection.count(), "chunk في قاعدة البيانات")