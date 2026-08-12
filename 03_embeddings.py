
from langchain_community.embeddings import HuggingFaceEmbeddings

# لاحظ: هنستخدم wrapper من LangChain حوالين نفس الموديل اللي جربناه
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# إنشاء قاعدة البيانات وتخزين الـ chunks فيها
