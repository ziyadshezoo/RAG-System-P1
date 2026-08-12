from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"  # هيستخدم الـ GPU تلقائياً
)

print("الموديل اتحمّل بنجاح")

#############################################

# Generation Function
def generate_answer(query, vectorstore, k=3):
    # 1. استرجاع أقرب chunks
    results = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in results])
    
    # 2. بناء الـ prompt
    prompt = f"""Context:
    {context}

    Question: {query}

    Answer based on the context above:"""
    
    # 3. تجهيز الرسالة بصيغة chat
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(text, return_tensors="pt").to(llm_model.device)
    
    # 4. توليد الإجابة
    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.3,
        do_sample=True
    )
    
    answer = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return answer

#######################################################
while True:
    question = input("Question or exit: ")
    if question.lower() == "exit":
        break
    answer = generate_answer(question, vectorstore)
    print("Answer:", answer)
    print("-" * 50)
