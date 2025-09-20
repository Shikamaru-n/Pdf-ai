# PDF Q&A App – Chat with Your PDFs

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot that allows you to **upload PDFs** and **query their content** using **GROQ LLM** and a **FAISS vector store**.  
It is built with **LangChain** and deployed via **Streamlit**.

---

## ✨ Features
- 📄 Upload and process **PDF documents**
- 🧠 Store and query embeddings in **FAISS** (local vector database)
- 🔍 Hybrid retrieval (dense + sparse BM25 with reranking via Cross-Encoder)
- 💬 Answer questions strictly from your documents (no outside knowledge)
- 🔒 API key management via `.env`
- 🛠️ Modular, well-commented code for easy extension
- 🎨 Simple Streamlit UI for chat-like experience

---

## ⚙️ Setup Instructions

### 1. Create a Virtual Environment

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up GROQ API Key

To use this RAG chatbot, you'll need a GROQ API key.

#### Get Your API Key

1. Visit [console.groq.com/keys](https://console.groq.com/keys)
2. Sign up or log in to your GROQ account
3. Create a new API key
4. Copy the generated API key

#### Set Environment Variable

**Windows (Command Prompt)**

```cmd
set GROQ_API_KEY=your_api_key_here
```

**Windows (PowerShell)**

```powershell
$env:GROQ_API_KEY="your_api_key_here"
```

**macOS / Linux**

```bash
export GROQ_API_KEY="your_api_key_here"
```

**Alternative: Create a .env file**

You can also create a `.env` file in the project root directory:

```
GROQ_API_KEY=your_api_key_here
```

> ⚠️ **Important**: Never commit your .env file to GitHub. Add it to your .gitignore.

---

## ▶️ Run the App

After setup, launch the Streamlit app:

```bash
streamlit run ragapp.py
```

If successful, the app will be available at:

```
http://localhost:8501
```

## 📂 Project Structure

```
Naive RAG Project/
│── ragapp.py          # Main streamlit app
│── requirements.txt   # Project dependencies
│── README.md          # Project documentation
│── .gitignore         # ignore faiss_db/, .env, temp/ files
│── .env               # Local API key storage (not pushed to Git)
│
├── Documents/         # Optional sample PDFs
├── helper/            # Helper scripts
```

> ⚠️ At runtime, a **faiss_db/** folder (FAISS vectore store) and a **temp/** folder will be created automatically. These are excluded from Git via .gitignore.