# retrieval
query = "What is the Pattern Recognition ?"

results = vectorstore.similarity_search(query, k=2) 

for i, doc in enumerate(results):
    print(f"Result {i+1}:\n")
    print(doc.page_content)
    print()