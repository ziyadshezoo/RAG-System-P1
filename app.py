%%writefile app.py
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.title("📄 RAG Document Q&A about the Topic 'pattern recognition' ")
st.write("Ask : questions about the topic 'pattern recognition' and get answers based on the content of the uploaded PDF document.")

@st.cache_resource
def load_everything():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    return vectorstore, tokenizer, llm_model

vectorstore, tokenizer, llm_model = load_everything()

def generate_answer(query, k=3):
    results = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in results])
    
    prompt = f"""Use the following context to answer the question. If the context doesn't contain the answer, say "I don't have enough information in the document."

Context:
{context}

Question: {query}

Answer:"""
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(llm_model.device)
    outputs = llm_model.generate(**inputs, max_new_tokens=300, do_sample=False)
    answer = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return answer, results

question = st.text_input("Ask a question about the topic 'pattern recognition':")

if st.button("Ask"):
    if question:
        with st.spinner("Thinking..."):
            answer, sources = generate_answer(question)
        st.markdown("### Answer:")
        st.write(answer)
        
        with st.expander("Sources used from the document"):
            for i, doc in enumerate(sources):
                st.write(f"**Source {i+1}:**")
                st.write(doc.page_content)
    else:
        st.warning("Please write the first question")

##################################################

# Ngrok

from pyngrok import ngrok

# حط الـ authtoken بتاعك هنا (من ngrok.com)
ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")

# شغّل streamlit في الخلفية
!streamlit run app.py &>/content/logs.txt &

# Colab Tunnel 
public_url = ngrok.connect(8501)
print("افتح الرابط ده:", public_url)