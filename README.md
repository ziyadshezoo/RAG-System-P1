# 📄 RAG Document Q&A System

A fully local, **100% free** Retrieval-Augmented Generation (RAG) pipeline that answers questions from your own PDF documents — no paid APIs, no external services required.

Built as a learning project to understand the core RAG pattern: chunking, embeddings, vector storage, retrieval, and generation.

---

## 🚀 How It Works

1. A PDF document is loaded and split into smaller text chunks.
2. Each chunk is converted into a numerical vector (embedding) that captures its meaning.
3. The embeddings are stored in a local vector database.
4. When a user asks a question, it's converted into an embedding and compared against the stored chunks to find the most relevant ones.
5. The relevant chunks are passed as context to a language model, which generates an answer grounded in the document — not from its own general knowledge.

```
PDF → Chunking → Embeddings → Vector DB (Chroma)
                                     ↓
User Question → Embedding → Similarity Search → Top-K Chunks
                                     ↓
                    Context + Question → LLM → Answer
```

---

## 🛠️ Tech Stack

| Component | Tool | Notes |
|---|---|---|
| Document Loading | `PyPDFLoader` (LangChain) | Loads and parses PDF files |
| Chunking | `RecursiveCharacterTextSplitter` (LangChain) | Splits text into overlapping chunks |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` | Runs locally, no API key needed |
| Vector Database | `Chroma` | Local, lightweight vector store |
| LLM (Generation) | `Qwen2.5-1.5B-Instruct` (Hugging Face `transformers`) | Runs locally on GPU |
| Interface | `Streamlit` | Simple web UI for asking questions |
| Public Access (Colab) | `pyngrok` | Exposes the local Streamlit app via a public URL |

**No paid API keys required anywhere in this project.**

---

## 📂 Project Structure

> Organized by pipeline step — each file corresponds to one stage of the RAG process.

```
rag-project/
│
├── 01_load_document.py        # Load PDF and inspect raw pages
├── 02_chunking.py             # Split document into chunks
├── 03_embeddings.py           # Test the embedding model
├── 04_vector_store.py         # Create Chroma DB and store chunk embeddings
├── 05_retrieval.py            # Test similarity search / retrieval
├── 06_generation.py           # Load LLM and generate answers using retrieved context
├── app.py                     # Streamlit interface (full pipeline)
├── requirements.txt           # Project dependencies
└── README.md                  # This file
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd rag-project
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your document
Place your PDF file in the project directory.

### 5. Run the pipeline scripts in order (01 → 06)
Each script can be run independently to understand that stage of the pipeline.

### 6. Launch the app
```bash
streamlit run app.py
```

---

## 🧪 Running on Google Colab

This project was originally built and tested on Google Colab with a free T4 GPU.

1. Enable GPU: `Runtime → Change runtime type → T4 GPU`
2. Install dependencies with `!pip install -q -r requirements.txt`
3. Upload your PDF using `google.colab.files.upload()`
4. Run the pipeline scripts/cells in order
5. Launch the Streamlit app and expose it publicly:
```python
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_TOKEN")

!streamlit run app.py &>/content/logs.txt &
public_url = ngrok.connect(8501)
print(public_url)
```

---

## 📌 Key Concepts Demonstrated

- **Chunking strategy** — fixed-size chunking with overlap to preserve context across chunk boundaries
- **Semantic embeddings** — converting text into vectors where similar meanings are numerically close
- **Vector similarity search** — retrieving the most relevant chunks for a given query
- **Grounded generation** — prompting the LLM to answer only from retrieved context, reducing hallucination
- **Language consistency** — prompt language should match the document's language for best results

---

## 🔮 Possible Improvements

- [ ] Support multiple document uploads
- [ ] Add source citations directly in the UI (already returned, can be made more visible)
- [ ] Experiment with different chunk sizes / overlap values
- [ ] Add an evaluation set of test questions (with/without answers in the document) to measure accuracy
- [ ] Try a larger LLM for improved answer quality
- [ ] Deploy without ngrok using Streamlit Community Cloud or Hugging Face Spaces

---

## 📝 License

This project is for educational purposes.
